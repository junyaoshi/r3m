import numpy as np
import pickle
import sys
from os.path import join
import cv2
import torch
import os
import matplotlib.pyplot as plt

def determine_which_hand(hand_info):
    left_hand_exists  = len(hand_info['pred_output_list'][0]['left_hand'])  > 0
    right_hand_exists = len(hand_info['pred_output_list'][0]['right_hand']) > 0
    if left_hand_exists and not right_hand_exists:
        return 'left_hand'
    if right_hand_exists and not left_hand_exists:
        return 'right_hand'
    if left_hand_exists and right_hand_exists:
        # select the hand with the bigger bounding box
        left_hand_bbox  = hand_info['hand_bbox_list'][0]['left_hand']
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
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def xywh_to_xyxy(xywh_bbox):
    x, y, w, h = xywh_bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return np.array([x1, y1, x2, y2])

def normalize_bbox(unnoramlized_bbox, img_size):
    img_x, img_y = img_size
    img_x, img_y = float(img_x), float(img_y)

    bbox_0 = unnoramlized_bbox[0] / img_x
    bbox_1 = unnoramlized_bbox[1] / img_y
    bbox_2 = unnoramlized_bbox[2] / img_x
    bbox_3 = unnoramlized_bbox[3] / img_y

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])

def unnormalize_bbox(normalized_bbox, img_size):
    img_x, img_y = img_size

    bbox_0 = int(normalized_bbox[0] * img_x)
    bbox_1 = int(normalized_bbox[1] * img_y)
    bbox_2 = int(normalized_bbox[2] * img_x)
    bbox_3 = int(normalized_bbox[3] * img_y)

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])

def mocap_path_to_rendered_path(mocap_path):
    path_split = mocap_path.split('/')
    path_split[-2] = 'rendered'
    path_split[-1] = path_split[-1].split('_')[0] + '.jpg'
    return join(*path_split)

def generate_single_visualization(
        current_hand_pose_path, future_hand_pose_path, future_cam,
        hand, pred_hand_bbox, pred_hand_pose,
        visualizer, hand_mocap, use_visualizer, device, run_on_cv_server
):
    from ss_utils.filter_utils import extract_pred_mesh_list
    from renderer.image_utils import draw_hand_bbox
    from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
    def render_bbox_and_hand_pose(
            img_original_bgr, hand_bbox_list, mesh_list
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

    def load_img_from_hand_info(hand_info):
        img_path = hand_info['image_path']
        if img_path[:8] == '/scratch' and run_on_cv_server:
            img_path = '/home' + img_path[8:]
        return cv2.imread(img_path)

    with open(current_hand_pose_path, 'rb') as f:
        current_hand_info = pickle.load(f)
    with open(future_hand_pose_path, 'rb') as f:
        future_hand_info = pickle.load(f)

    current_hand_bbox_list, current_mesh_list = extract_hand_bbox_and_mesh_list(
        current_hand_info, hand
    )
    current_img = load_img_from_hand_info(current_hand_info)
    current_rendered_img = render_bbox_and_hand_pose(
            current_img, current_hand_bbox_list, current_mesh_list
    )

    future_hand_bbox_list, future_mesh_list = extract_hand_bbox_and_mesh_list(
        future_hand_info, hand
    )
    future_img = load_img_from_hand_info(future_hand_info)
    future_rendered_img = render_bbox_and_hand_pose(
        future_img, future_hand_bbox_list, future_mesh_list
    )

    img_h, img_w, _ = future_img.shape
    pred_hand_bbox = unnormalize_bbox(pred_hand_bbox, (img_w, img_h))
    pred_hand_bbox[2] = min(pred_hand_bbox[2], img_w - pred_hand_bbox[0])
    pred_hand_bbox[3] = min(pred_hand_bbox[3], img_h - pred_hand_bbox[1])
    pred_hand_bbox_list = future_hand_bbox_list.copy()
    pred_hand_bbox_list[0][hand] = pred_hand_bbox
    future_hand_betas = future_hand_info['pred_output_list'][0][hand]['pred_hand_betas']

    #  get predicted smpl verts and joints,
    if hand == 'left_hand':
        pred_hand_pose[1::3] *= -1
        pred_hand_pose[2::3] *= -1
    pred_verts, _ = hand_mocap.model_regressor.get_smplx_output(
        torch.Tensor(pred_hand_pose).unsqueeze(0).to(device),
        torch.Tensor(future_hand_betas).to(device)
    )

    # Convert vertices into bbox & image space
    pred_verts_origin = pred_verts.detach().cpu().numpy()[:, hand_mocap.model_regressor.right_hand_verts_idx, :][0]
    if hand == 'left_hand':
        pred_verts_origin[:, 0] *= -1
    faces = future_hand_info['pred_output_list'][0][hand]['faces']
    cam_scale = future_cam[0]
    cam_trans = future_cam[1:]
    vert_smplcoord = pred_verts_origin.copy()
    vert_bboxcoord = convert_smpl_to_bbox(
        vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
    )  # SMPL space -> bbox space
    *_, bbox_scale_ratio, bbox_processed = hand_mocap.process_hand_bbox(
        future_img, pred_hand_bbox, hand, add_margin=True
    )
    bbox_top_left = np.array(bbox_processed[:2])
    vert_imgcoord = convert_bbox_to_oriIm(
        vert_bboxcoord, bbox_scale_ratio, bbox_top_left,
        future_img.shape[1], future_img.shape[0]
    )
    pred_hand_info = future_hand_info.copy()
    pred_hand_info['pred_output_list'][0][hand]['pred_vertices_img'] = vert_imgcoord
    pred_hand_info['pred_output_list'][0][hand]['faces'] = faces

    _, pred_mesh_list = extract_hand_bbox_and_mesh_list(
        pred_hand_info, hand
    )
    pred_rendered_img = render_bbox_and_hand_pose(
        future_img, pred_hand_bbox_list, pred_mesh_list
    )

    vis_img_bgr = np.vstack([current_rendered_img, future_rendered_img, pred_rendered_img])
    vis_img = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)
    return vis_img.astype(np.uint8)

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
    # # test path conversion
    # mocap_path = '/home/junyao/Datasets/something_something_processed/' \
    #              'push_left_right/train/mocap_output/0/mocap/frame10_prediction_result.pkl'
    # rendered_path = mocap_path_to_rendered_path(mocap_path)
    # print(rendered_path)

    # # test bbox normalization
    # future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
    #                         'push_left_right/valid/mocap_output/108375/mocap/frame31_prediction_result.pkl'
    # hand= 'left_hand'
    # with open(future_hand_pose_path, 'rb') as f:
    #     future_hand_info = pickle.load(f)
    # img_path = future_hand_info['image_path']
    # if img_path[:8] == '/scratch':
    #     img_path = '/home' + img_path[8:]
    # img = cv2.imread(img_path)
    # future_hand_bbox = future_hand_info['hand_bbox_list'][0][hand]
    # img_size = (img.shape[1], img.shape[0])
    # norm_future_hand_bbox = normalize_bbox(future_hand_bbox, img_size)
    # unnorm_future_hand_bbox = unnormalize_bbox(norm_future_hand_bbox, img_size)
    # print(f'Original bbox: {future_hand_bbox}')
    # print(f'Normalized bbox: {norm_future_hand_bbox}')
    # print(f'Unnormalized bbox: {unnorm_future_hand_bbox}')

    # test visualization
    use_visualizer = True
    frankmocap_path = '/home/junyao/LfHV/frankmocap'
    r3m_path ='/home/junyao/LfHV/r3m'
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

    current_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                             'push_left_right/valid/mocap_output/80757/mocap/frame24_prediction_result.pkl'
    future_hand_pose_path = '/home/junyao/Datasets/something_something_processed/' \
                            'push_left_right/valid/mocap_output/80757/mocap/frame29_prediction_result.pkl'
    hand = 'right_hand'

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

    print('Begin visualization.')
    vis_img = generate_single_visualization(
        current_hand_pose_path=current_hand_pose_path,
        future_hand_pose_path=future_hand_pose_path,
        future_cam=future_camera,
        hand=hand,
        pred_hand_bbox=future_hand_bbox,
        pred_hand_pose=future_hand_pose,
        visualizer=visualizer,
        hand_mocap=hand_mocap,
        use_visualizer=use_visualizer,
        device='cuda',
        run_on_cv_server=True
    )

    plt.figure()
    plt.imshow(vis_img)
    plt.savefig('vis_img.png')
    # plt.show()
    plt.close()
