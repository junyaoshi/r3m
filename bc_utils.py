import numpy as np
import pickle
import sys
from os.path import join
import cv2
import torch
import os
import matplotlib.pyplot as plt
from copy import deepcopy

CV_TASKS = [
    'push_left',
    'push_right',
    'move_down',
    'move_up',
]
CLUSTER_TASKS = [
    'move_away',
    'move_towards',
    'move_down',
    'move_up',
    'pull_left',
    'pull_right',
    'push_left',
    'push_right',
    'push_slightly'
]


def cv_task_to_cluster_task(cv_task):
    """Convert one-hot cv task to one-hot cluster task"""
    task_name = CV_TASKS[torch.argmax(cv_task)]
    cluster_task_idx = CLUSTER_TASKS.index(task_name)
    cluster_task = torch.zeros(len(CLUSTER_TASKS))
    cluster_task[cluster_task_idx] = 1
    return cluster_task


def cluster_task_to_cv_task(cluster_task):
    """Convert one-hot cv task to one-hot cluster task"""
    task_name = CLUSTER_TASKS[torch.argmax(cluster_task)]
    cv_task_idx = CV_TASKS.index(task_name)
    cv_task = torch.zeros(len(CV_TASKS))
    cv_task[cv_task_idx] = 1
    return cv_task


def determine_which_hand(hand_info):
    left_hand_exists = len(hand_info['pred_output_list'][0]['left_hand']) > 0
    right_hand_exists = len(hand_info['pred_output_list'][0]['right_hand']) > 0
    if left_hand_exists and not right_hand_exists:
        return 'left_hand'
    if right_hand_exists and not left_hand_exists:
        return 'right_hand'
    if left_hand_exists and right_hand_exists:
        # select the hand with the bigger bounding box
        left_hand_bbox = hand_info['hand_bbox_list'][0]['left_hand']
        *_, lw, lh = left_hand_bbox
        right_hand_bbox = hand_info['hand_bbox_list'][0]['right_hand']
        *_, rw, rh = right_hand_bbox
        if lw * lh >= rw * rh:
            return 'left_hand'
        else:
            return 'right_hand'
    else:
        raise ValueError('No hand detected!')


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def xywh_to_xyxy(xywh_bbox):
    x, y, w, h = xywh_bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return np.array([x1, y1, x2, y2])


def normalize_bbox(unnormalized_bbox, img_size):
    img_x, img_y = img_size
    img_x, img_y = float(img_x), float(img_y)

    bbox_0 = unnormalized_bbox[0] / img_x
    bbox_1 = unnormalized_bbox[1] / img_y
    bbox_2 = unnormalized_bbox[2] / img_x
    bbox_3 = unnormalized_bbox[3] / img_y

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])


def unnormalize_bbox(normalized_bbox, img_size):
    img_x, img_y = img_size

    bbox_0 = int(round(normalized_bbox[0] * img_x))
    bbox_1 = int(round(normalized_bbox[1] * img_y))
    bbox_2 = int(round(normalized_bbox[2] * img_x))
    bbox_3 = int(round(normalized_bbox[3] * img_y))

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])


def unnormalize_bbox_batch(normalized_bbox, img_size):
    img_x, img_y = img_size

    bbox_0 = torch.mul(normalized_bbox[:, 0], img_x).round().int()
    bbox_1 = torch.mul(normalized_bbox[:, 1], img_y).round().int()
    bbox_2 = torch.mul(normalized_bbox[:, 2], img_x).round().int()
    bbox_3 = torch.mul(normalized_bbox[:, 3], img_y).round().int()

    return torch.stack([bbox_0, bbox_1, bbox_2, bbox_3]).swapaxes(0, 1)


def mocap_path_to_rendered_path(mocap_path):
    path_split = mocap_path.split('/')
    path_split[-2] = 'rendered'
    path_split[-1] = path_split[-1].split('_')[0] + '.jpg'
    return join(*path_split)


def _convert_smpl_to_bbox(data_z, scale):
    resnet_input_size_half = 224 * 0.5
    data_z = torch.mul(data_z, scale.unsqueeze(1)) # apply scaling
    data_z *= resnet_input_size_half  # 112 is originated from hrm's input size (224,24)
    return data_z


def _convert_bbox_to_oriIm(data_z, img_shape, hand_bbox, final_size=224):
    ori_height, ori_width = img_shape[:, 0], img_shape[:, 1]
    hand_bbox = unnormalize_bbox_batch(hand_bbox, (ori_width, ori_height))
    hand_bbox[:, 2] = torch.min(hand_bbox[:, 2], ori_width - hand_bbox[:, 0])
    hand_bbox[:, 3] = torch.min(hand_bbox[:, 3], ori_height - hand_bbox[:, 1])

    min_x, min_y = hand_bbox[:, 0], hand_bbox[:, 1]
    width, height = hand_bbox[:, 2], hand_bbox[:, 3]
    max_x = min_x + width
    max_y = min_y + height

    for i in range(width.size(0)):
        if width[i] > height[i]:
            margin = torch.div((width[i] - height[i]), 2, rounding_mode='floor')
            min_y[i] = torch.max(min_y[i] - margin, torch.zeros_like(margin))
            max_y[i] = torch.min(max_y[i] + margin, ori_height[i])
        else:
            margin = torch.div((height[i] - width[i]), 2, rounding_mode='floor')
            min_x[i] = torch.max(min_x[i] - margin, torch.zeros_like(margin))
            max_x[i] = torch.min(max_x[i] + margin, ori_width[i])

    margin = (0.3 * (max_y - min_y)).round().int()  # if use loose crop, change 0.3 to 1.0
    min_y = torch.max(min_y - margin, torch.zeros_like(margin))
    max_y = torch.min(max_y + margin, ori_height)
    min_x = torch.max(min_x - margin, torch.zeros_like(margin))
    max_x = torch.min(max_x + margin, ori_width)

    new_size = torch.max(max_x - min_x, max_y - min_y)
    bbox_ratio = final_size / new_size
    data_z = torch.div(data_z, bbox_ratio.unsqueeze(1))

    return data_z


def load_img_from_hand_info(hand_info, robot_demos, run_on_cv_server):
    img_path = hand_info['image_path']
    if img_path[:8] == '/scratch' and run_on_cv_server:
        img_path = '/home' + img_path[8:]

    if robot_demos:
        # replace frames with robot_frames
        img_path = '/' + join(*list(map(lambda x: x.replace('frames', 'robot_frames'), img_path.split('/'))))
    return cv2.imread(img_path)


def generate_single_visualization(
        current_hand_pose_path,
        future_hand_pose_path,
        future_cam,
        hand,
        pred_hand_bbox,
        pred_hand_pose,
        pred_hand_shape,
        task_names,
        task,
        visualizer,
        hand_mocap,
        use_visualizer,
        run_on_cv_server,
        current_img=None,
        future_img=None,
        original_task=False,
        robot_demos=False,
        log_depth=False,
        current_depth=None,
        future_depth=None,
        pred_depth=None
):
    from ss_utils.filter_utils import extract_pred_mesh_list
    from renderer.image_utils import draw_hand_bbox
    from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
    def render_bbox_and_hand_pose(
            visualizer, img_original_bgr, hand_bbox_list, mesh_list, use_visualizer
    ):
        res_img = img_original_bgr.copy()
        res_img = draw_hand_bbox(res_img, hand_bbox_list)
        res_img = visualizer.render_pred_verts(res_img, mesh_list) if use_visualizer else np.zeros_like(res_img)
        return res_img

    def extract_hand_bbox_and_mesh_list(hand_info, hand):
        other_hand = 'left_hand' if hand == 'right_hand' else 'right_hand'
        hand_bbox_list = hand_info['hand_bbox_list']
        hand_bbox_list[0][other_hand] = None
        hand_info['pred_output_list'][0][other_hand] = None
        mesh_list = extract_pred_mesh_list(hand_info)
        return hand_bbox_list, mesh_list

    with open(current_hand_pose_path, 'rb') as f:
        current_hand_info = pickle.load(f)
    current_hand_bbox_list, current_mesh_list = extract_hand_bbox_and_mesh_list(
        current_hand_info, hand
    )
    if current_img is None:
        current_img = load_img_from_hand_info(current_hand_info, robot_demos, run_on_cv_server)
    current_rendered_img = render_bbox_and_hand_pose(
        visualizer, current_img, current_hand_bbox_list, current_mesh_list, use_visualizer
    )

    img_h, img_w, _ = current_img.shape
    pred_hand_bbox = unnormalize_bbox(pred_hand_bbox.detach().cpu().numpy(), (img_w, img_h))
    pred_hand_bbox[2] = min(pred_hand_bbox[2], img_w - pred_hand_bbox[0])
    pred_hand_bbox[3] = min(pred_hand_bbox[3], img_h - pred_hand_bbox[1])
    pred_hand_bbox_list = deepcopy(current_hand_bbox_list)
    pred_hand_bbox_list[0][hand] = pred_hand_bbox

    #  get predicted smpl verts and joints,
    if hand == 'left_hand':
        pred_hand_pose[1::3] *= -1
        pred_hand_pose[2::3] *= -1
    pred_verts, _ = hand_mocap.model_regressor.get_smplx_output(
        pred_hand_pose.unsqueeze(0),
        pred_hand_shape.unsqueeze(0)
    )

    # Convert vertices into bbox & image space
    pred_verts_origin = pred_verts.detach().cpu().numpy()[:, hand_mocap.model_regressor.right_hand_verts_idx, :][0]
    if hand == 'left_hand':
        pred_verts_origin[:, 0] *= -1
    cam_scale = future_cam[0]
    cam_trans = future_cam[1:]
    vert_smplcoord = pred_verts_origin.copy()
    vert_bboxcoord = convert_smpl_to_bbox(
        vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
    )  # SMPL space -> bbox space
    *_, bbox_scale_ratio, bbox_processed = hand_mocap.process_hand_bbox(
        current_img, pred_hand_bbox, hand, add_margin=True
    )
    bbox_top_left = np.array(bbox_processed[:2])
    vert_imgcoord = convert_bbox_to_oriIm(
        vert_bboxcoord, bbox_scale_ratio, bbox_top_left,
        img_w, img_h
    )
    pred_hand_info = deepcopy(current_hand_info)
    pred_hand_info['pred_output_list'][0][hand]['pred_vertices_img'] = vert_imgcoord

    _, pred_mesh_list = extract_hand_bbox_and_mesh_list(
        pred_hand_info, hand
    )

    no_future_info = future_hand_pose_path is None
    if no_future_info:
        pred_rendered_img = render_bbox_and_hand_pose(
            visualizer, current_img,
            pred_hand_bbox_list, pred_mesh_list, use_visualizer
        )
        vis_img_bgr = np.vstack([current_rendered_img, pred_rendered_img])
    else:
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_hand_bbox_list, future_mesh_list = extract_hand_bbox_and_mesh_list(
            future_hand_info, hand
        )
        if future_img is None:
            future_img = load_img_from_hand_info(future_hand_info, robot_demos, run_on_cv_server)
        future_rendered_img = render_bbox_and_hand_pose(
            visualizer, future_img, future_hand_bbox_list, future_mesh_list, use_visualizer
        )
        pred_rendered_img = render_bbox_and_hand_pose(
            visualizer, future_img,
            pred_hand_bbox_list, pred_mesh_list, use_visualizer
        )
        vis_img_bgr = np.vstack([current_rendered_img, pred_rendered_img, future_rendered_img])

    if log_depth:
        assert current_depth is not None and future_depth is not None and pred_depth is not None
        white = np.zeros((200, vis_img_bgr.shape[1], 3), np.uint8)
    else:
        white = np.zeros((50, vis_img_bgr.shape[1], 3), np.uint8)
    white[:] = (255, 255, 255)
    vis_img_bgr = cv2.vconcat((white, vis_img_bgr))
    font = cv2.FONT_HERSHEY_COMPLEX
    task_name = task_names[np.argmax(task)]
    task_str = "Original Task" if original_task else "Task"
    task_desc_str = f'{task_str}: {task_name}'
    cv2.putText(vis_img_bgr, task_desc_str, (30, 40), font, 0.6, (0, 0, 0), 2, 0)
    if log_depth:
        current_depth_str = f'Current Depth: {current_depth:.4f}'
        cv2.putText(vis_img_bgr, current_depth_str, (30, 90), font, 0.6, (0, 0, 0), 2, 0)
        pred_depth_str = f'Predicted Depth: {pred_depth:.4f}'
        cv2.putText(vis_img_bgr, pred_depth_str, (30, 140), font, 0.6, (0, 0, 0), 2, 0)
        if not no_future_info:
            future_depth_str = f'Future Depth: {future_depth:.4f}'
            cv2.putText(vis_img_bgr, future_depth_str, (30, 190), font, 0.6, (0, 0, 0), 2, 0)

    vis_img = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)
    return vis_img.astype(np.uint8)


def pose_to_joint_depth(
        hand_mocap,
        hand, pose, bbox, cam, img_shape,
        device,
        shape, shape_path=None
):
    assert not (shape is None and shape_path is None)
    if shape is None:
        shape = []
        for i, p in enumerate(shape_path):
            with open(p, 'rb') as f:
                hand_info = pickle.load(f)
            hand_shape = hand_info['pred_output_list'][0][hand[i]]['pred_hand_betas']
            shape.append(hand_shape)
        shape = np.vstack(shape)
        shape = torch.Tensor(shape).to(device)

    _, joints_smplcoord = hand_mocap.model_regressor.get_smplx_output(pose, shape)
    joints_smplcoord_z = joints_smplcoord[:, :, 2]
    cam_scale = cam[:, 0]
    joints_bboxcoord_z = _convert_smpl_to_bbox(joints_smplcoord_z, cam_scale)  # SMPL space -> bbox space
    joints_imgcoord_z = _convert_bbox_to_oriIm(joints_bboxcoord_z, img_shape, bbox)
    return joints_imgcoord_z


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


if __name__ == '__main__':
    test_path_conversion = False
    test_visualization = False
    test_pose_to_joint_z = False
    test_robot_visualization = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_visualizer = True
    no_future_info = False
    log_tb = False

    if test_path_conversion:
        # test path conversion
        mocap_path = '/home/junyao/Datasets/something_something_processed/' \
                     'push_left_right/train/mocap_output/0/mocap/frame10_prediction_result.pkl'
        rendered_path = mocap_path_to_rendered_path(mocap_path)
        print(rendered_path)

        # test bbox normalization
        future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                                'push_left_right/valid/mocap_output/108375/mocap/frame31_prediction_result.pkl'
        hand= 'left_hand'
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        img_path = future_hand_info['image_path']
        if img_path[:8] == '/scratch':
            img_path = '/home' + img_path[8:]
        img = cv2.imread(img_path)
        future_hand_bbox = future_hand_info['hand_bbox_list'][0][hand]
        img_size = (img.shape[1], img.shape[0])
        norm_future_hand_bbox = normalize_bbox(future_hand_bbox, img_size)
        unnorm_future_hand_bbox = unnormalize_bbox(norm_future_hand_bbox, img_size)
        print(f'Original bbox: {future_hand_bbox}')
        print(f'Normalized bbox: {norm_future_hand_bbox}')
        print(f'Unnormalized bbox: {unnorm_future_hand_bbox}')

    if test_visualization:
        # test visualization
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        from handmocap.hand_mocap_api import HandMocap

        if use_visualizer:
            print('Loading opengl visualizer.')
            from renderer.visualizer import Visualizer

            visualizer = Visualizer('opengl')
        else:
            visualizer = None
        print('Loading frank mocap hand module.')
        checkpoint_hand = join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device='cuda')
        os.chdir(r3m_path)

        current_hand_pose_path = '/home/junyao/Datasets/something_something_hand_demos_same_hand/' \
                                 'move_up/mocap_output/1/mocap/frame0_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_hand_demos_same_hand/' \
                                'move_up/mocap_output/1/mocap/frame1_prediction_result.pkl'
        hand = 'left_hand'

        print('Preprocessing hand data from pkl.')
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch':
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        future_hand_bbox = normalize_bbox(
            future_hand_info['hand_bbox_list'][0][hand],
            (future_image.shape[1], future_image.shape[0])
        )
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)

        task_names = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
            'push_slightly',
            'push_left_right'
        ]
        task = np.zeros(len(task_names))
        task[-1] = 1

        print('Begin visualization.')
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path,
            future_hand_pose_path=None if no_future_info else future_hand_pose_path,
            future_cam=future_camera,
            hand=hand,
            pred_hand_bbox=torch.from_numpy(future_hand_bbox).to(device),
            pred_hand_pose=torch.from_numpy(future_hand_pose).to(device),
            pred_hand_shape=torch.from_numpy(future_hand_shape).to(device),
            task_names=task_names,
            task=task,
            visualizer=visualizer,
            hand_mocap=hand_mocap,
            use_visualizer=use_visualizer,
            run_on_cv_server=True,
            log_depth=True,
            current_depth=1,
            future_depth=future_joint_depth.mean(),
            pred_depth=1
        )

        plt_path = 'vis_img.png'
        plt.figure()
        plt.imshow(vis_img)
        plt.savefig('vis_img.png')
        # plt.show()
        plt.close()
        print(f'Saved visualization image to {plt_path}.')

        if log_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir='tb_vis_test')
            writer.add_scalar('loss', 3, 1)
            writer.add_image(f'vis_images', vis_img, 1, dataformats='HWC')
            print(f'Saved tensorboard visualization image to tb_vis_test.')

    if test_pose_to_joint_z:
        # test pose to joint z
        run_on_cv_server = True
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        device = 'cuda'
        from handmocap.hand_mocap_api import HandMocap

        print('Loading frank mocap hand module.')
        checkpoint_hand = join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=2)
        os.chdir(r3m_path)

        current_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                                 'push_left_right/valid/mocap_output/80757/mocap/frame24_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                                'push_left_right/valid/mocap_output/80757/mocap/frame29_prediction_result.pkl'

        hand = ['right_hand', 'right_hand']
        pose_path = [current_hand_pose_path, future_hand_pose_path]

        print('Preprocessing hand data from pkl.')
        with open(current_hand_pose_path, 'rb') as f:
            current_hand_info = pickle.load(f)
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)

        current_hand_pose = current_hand_info['pred_output_list'][0]['right_hand']['pred_hand_pose']
        future_hand_pose = future_hand_info['pred_output_list'][0]['right_hand']['pred_hand_pose']
        pose = np.vstack([current_hand_pose, future_hand_pose])
        pose = torch.Tensor(pose).to(device)

        current_hand_shape = current_hand_info['pred_output_list'][0]['right_hand']['pred_hand_betas']
        future_hand_shape = future_hand_info['pred_output_list'][0]['right_hand']['pred_hand_betas']
        shape = np.vstack([current_hand_shape, future_hand_shape])
        shape = torch.Tensor(shape).to(device)

        current_cam = current_hand_info['pred_output_list'][0]['right_hand']['pred_camera']
        future_cam = future_hand_info['pred_output_list'][0]['right_hand']['pred_camera']
        cam = np.vstack([current_cam, future_cam])
        cam = torch.Tensor(cam).to(device)

        current_hand_bbox = current_hand_info['hand_bbox_list'][0]['right_hand']
        current_image_path = current_hand_info['image_path']
        if current_image_path[:8] == '/scratch' and run_on_cv_server:
            current_image_path = '/home' + current_image_path[8:]
        current_image = cv2.imread(current_image_path)
        current_hand_bbox = normalize_bbox(current_hand_bbox, (current_image.shape[1], current_image.shape[0]))
        current_img_shape = current_image.shape
        future_hand_bbox = future_hand_info['hand_bbox_list'][0]['right_hand']
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch' and run_on_cv_server:
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_bbox = normalize_bbox(future_hand_bbox, (future_image.shape[1], future_image.shape[0]))
        future_img_shape = future_image.shape
        hand_bbox = np.vstack([current_hand_bbox, future_hand_bbox])
        hand_bbox = torch.Tensor(hand_bbox).to(device)
        img_shape = np.vstack([current_img_shape, future_img_shape])
        img_shape = torch.Tensor(img_shape).to(device).round().int()

        joint_z = pose_to_joint_depth(
            hand_mocap, hand, pose, hand_bbox, cam, img_shape, device,
            shape=None, shape_path=pose_path
        )

        gt_current_joint_z = current_hand_info['pred_output_list'][0]['right_hand']['pred_joints_img'][:, 2]
        gt_future_joint_z = future_hand_info['pred_output_list'][0]['right_hand']['pred_joints_img'][:, 2]
        gt_joint_z = np.vstack([gt_current_joint_z, gt_future_joint_z])
        gt_joint_z = torch.Tensor(gt_joint_z).to(device)

        print(f'joint_z: \n{joint_z}')
        print(f'gt_joint_z: \n{gt_joint_z}')

    if test_robot_visualization:
        # test visualization
        frankmocap_path = '/home/junyao/LfHV/frankmocap'
        r3m_path = '/home/junyao/LfHV/r3m'
        sys.path.insert(1, frankmocap_path)
        os.chdir(frankmocap_path)
        from handmocap.hand_mocap_api import HandMocap

        if use_visualizer:
            print('Loading opengl visualizer.')
            from renderer.visualizer import Visualizer

            visualizer = Visualizer('opengl')
        else:
            visualizer = None
        print('Loading frank mocap hand module.')
        checkpoint_hand = join(
            frankmocap_path,
            'extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
        )
        smpl_dir = join(frankmocap_path, 'extra_data/smpl')
        hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device='cuda')
        os.chdir(r3m_path)

        current_hand_pose_path = '/home/junyao/Datasets/something_something_robot_demos/' \
                                 'push_right/mocap_output/1/mocap/frame2_prediction_result.pkl'
        future_hand_pose_path = '/home/junyao/Datasets/something_something_robot_demos/' \
                                'push_right/mocap_output/1/mocap/frame3_prediction_result.pkl'
        hand = 'left_hand'

        print('Preprocessing hand data from pkl.')
        with open(future_hand_pose_path, 'rb') as f:
            future_hand_info = pickle.load(f)
        future_image_path = future_hand_info['image_path']
        if future_image_path[:8] == '/scratch':
            future_image_path = '/home' + future_image_path[8:]
        future_image = cv2.imread(future_image_path)
        future_hand_pose = future_hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        future_hand_bbox = normalize_bbox(
            future_hand_info['hand_bbox_list'][0][hand],
            (future_image.shape[1], future_image.shape[0])
        )
        future_camera = future_hand_info['pred_output_list'][0][hand]['pred_camera']
        future_joint_depth = future_hand_info['pred_output_list'][0][hand]['pred_joints_img'][:, 2]
        future_hand_shape = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas'].reshape(10)

        task_names = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
            'push_slightly',
            'push_left_right'
        ]
        task = np.zeros(len(task_names))
        task[-1] = 1

        print('Begin visualization.')
        vis_img = generate_single_visualization(
            current_hand_pose_path=current_hand_pose_path,
            future_hand_pose_path=None if no_future_info else future_hand_pose_path,
            future_cam=future_camera,
            hand=hand,
            pred_hand_bbox=torch.from_numpy(future_hand_bbox).to(device),
            pred_hand_pose=torch.from_numpy(future_hand_pose).to(device),
            pred_hand_shape=torch.from_numpy(future_hand_shape).to(device),
            task_names=task_names,
            task=task,
            visualizer=visualizer,
            hand_mocap=hand_mocap,
            use_visualizer=use_visualizer,
            run_on_cv_server=True,
            robot_demos=True,
            log_depth=True,
            current_depth=1,
            future_depth=future_joint_depth.mean(),
            pred_depth=1
        )

        plt_path = 'vis_img.png'
        plt.figure()
        plt.imshow(vis_img)
        plt.savefig('vis_img.png')
        # plt.show()
        plt.close()
        print(f'Saved visualization image to {plt_path}.')

        if log_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir='tb_vis_test')
            writer.add_scalar('loss', 3, 1)
            writer.add_image(f'vis_images', vis_img, 1, dataformats='HWC')
            print(f'Saved tensorboard visualization image to tb_vis_test.')
