import argparse
import time
from tqdm import tqdm
from datetime import timedelta

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import AgentTransferable
from utils.bc_utils import *
from utils.data_utils import ALL_TASKS, load_pkl, get_sum_val, get_target_val, get_res_val
from bc_models.resnet import EndtoEndNet
from utils.vis_utils import generate_transferable_visualization


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training BC network.')

    # model
    parser.add_argument("--n_blocks", type=int, default=8,
                        help="number of 2-layer blocks in the network")
    parser.add_argument("--net_type", type=str, default="residual",
                        choices=[
                            'mlp',  # multilayer perceptrons without residual connections
                            'residual'  # residual network
                        ],
                        help="network architecture to use")

    # data
    parser.add_argument('--time_interval', type=int, default=15,
                        help='how many frames into the future to predict')
    parser.add_argument('--iou_thresh', type=float, default=0.7,
                        help='IoU threshold for filtering the data')
    parser.add_argument("--depth_descriptor", type=str, default="scaling_factor",
                        choices=['scaling_factor', 'normalized_bbox_size'],
                        help="descriptor for estimating hand depth")
    parser.add_argument('--no_shuffle', action='store_true',
                        help='if true, dataloader is not shuffled')
    parser.add_argument("--stage", type=str, default="all",
                        choices=[
                            'all',  # use all data
                            'pre',  # use pre-interaction data
                            'during' # use during-interaction data
                        ],
                        help="stage used to filter the dataset")

    # training and evaluation
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='weight for xy loss')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='weight for depth loss')
    parser.add_argument('--lambda3', type=float, default=1.0,
                        help='weight for orientation loss')
    parser.add_argument('--lambda4', type=float, default=1.0,
                        help='weight for contact loss')
    parser.add_argument('--bce_weight_mult', type=float, default=1.0,
                        help='multiplier for BCE loss pos_weight parameter')
    parser.add_argument('--pred_mode', type=str, default="residual",
                        choices=[
                            'original', # use original target values
                            'residual'  # use residual values (target - current)
                        ],
                        help='prediction mode for the model')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='perform evaluation after this many epochs')
    parser.add_argument('--log_scalar_freq', type=int, default=100,
                        help='log scalars after this many steps')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save model after this many epochs')
    parser.add_argument('--vis_freq', type=int, default=2000,
                        help='visualize rendered images after this many steps')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Evaluate model on training set instead of validation set (for debugging)')
    parser.add_argument('--eval_tasks', action='store_true',
                        help='if true, perform task-conditioned evaluatuation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--vis_sample_size', type=int, default=5,
                        help='number of samples to visualize on tensorboard, set to 0 to disable')
    parser.add_argument('--task_vis_sample_size', type=int, default=2,
                        help='number of task-conditioned samples to visualize on tensorboard, '
                             'set to 0 to disable task-conditioned visualization')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloaders')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run tasks on cv-server; else, run all tasks on cluster')
    parser.add_argument('--use_visualizer', action='store_true',
                        help='if true, use opengl visualizer to render results and show on tensorboard')

    # debugging and sanity check
    parser.add_argument('--sanity_check', action='store_true', default=False,
                        help='perform sanity check (try to only fit a few examples)')
    parser.add_argument('--sanity_check_size', type=int, default=32,
                        help='number of data for sanity check')
    parser.add_argument('--debug', action='store_true',
                        help='if true, enter debug mode, load 50 videos and no parallel workers')

    # paths
    parser.add_argument('--root', type=str, default='/home/junyao/LfHV/r3m/transferable_bc_ckpts',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='debug',
                        help='id used for storing intermediate results')
    parser.add_argument('--data_home_dirs', nargs='+',
                        default='/home/junyao/Datasets/something_something_processed',
                        help='list of locations of the data corpus, example: dir1 dir2 dir3')
    parser.add_argument('--frankmocap_path', type=str,
                        default='/home/junyao/LfHV/frankmocap',
                        help='location of frank mocap')
    parser.add_argument('--r3m_path', type=str,
                        default='/home/junyao/LfHV/r3m',
                        help='location of R3M')
    parser.add_argument('--depth_norm_params_path', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils/depth_normalization_params.pkl',
                        help='location of depth normalization params')
    parser.add_argument('--ori_norm_params_path', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils/ori_normalization_params.pkl',
                        help='location of orientation normalization params')
    parser.add_argument('--contact_count_dir', type=str,
                        default='/home/junyao/LfHV/frankmocap/ss_utils',
                        help='location of orientation normalization params')

    args = parser.parse_args()
    return args


def main(args):
    program_start = time.time()
    if args.sanity_check:
        args.no_shuffle = True
        args.eval_on_train = True
        args.log_scalar_freq = 1
        torch.manual_seed(5157)
        np.random.seed(5157)
        assert args.batch_size <= args.sanity_check_size
    args.pred_residual = args.pred_mode == 'residual'
    assert args.vis_sample_size <= args.batch_size
    args.save = join(args.root, args.save)
    assert (args.lambda1 > 0) and (args.lambda2 > 0) and (args.lambda3 > 0) and (args.lambda4 > 0)
    task_names = ALL_TASKS
    args.task_names = task_names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}.')
    writer = SummaryWriter(log_dir=args.save, flush_secs=60)

    # visualizer and frank mocap
    print('Loading frankmocap visualizer...')
    sys.path.insert(1, args.frankmocap_path)
    os.chdir(args.frankmocap_path)
    from renderer.visualizer import Visualizer
    from handmocap.hand_mocap_api import HandMocap
    visualizer = Visualizer('opengl') if args.use_visualizer else None
    checkpoint_hand = join(args.frankmocap_path, 'extra_data/hand_module/pretrained_weights/pose_shape_best.pth')
    smpl_dir = join(args.frankmocap_path, 'extra_data/smpl')
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device, batch_size=1)
    os.chdir(args.r3m_path)
    print('Visualizer loaded.')

    # compute dimensions
    args.r3m_dim, args.task_dim = 2048, len(task_names)
    args.xy_dim, args.depth_dim, args.ori_dim, args.contact_dim = 2, 1, 3, 1
    args.input_dim = sum([args.r3m_dim, args.task_dim, args.xy_dim, args.depth_dim, args.ori_dim, args.contact_dim])
    args.output_dim = sum([args.xy_dim, args.depth_dim, args.ori_dim, args.contact_dim])

    # log args
    pprint(f'args: \n{args}')
    for arg in vars(args):
        writer.add_text(f'args/{str(arg)}', str(getattr(args, arg)))

    # initialize model
    model = EndtoEndNet(
        in_features=args.input_dim,
        out_features=args.output_dim,
        dims=(),
        n_blocks=args.n_blocks,
        residual=args.net_type == 'residual'
    ).to(device).float()
    print(f'Loaded transferable BC model. Blocks: {args.n_blocks}, Network: {args.net_type}.')
    print(f'param size = {count_parameters_in_M(model)}M')

    # load pkl
    args.depth_norm_params = load_pkl(args.depth_norm_params_path)[args.depth_descriptor]
    args.ori_norm_params = load_pkl(args.ori_norm_params_path)
    args.contact_count_path = join(args.contact_count_dir, f'contact_count_t={args.time_interval}.pkl')
    args.contact_count = load_pkl(args.contact_count_path)
    args.bce_pos_weight = args.bce_weight_mult * args.contact_count['n_neg'] / args.contact_count['n_pos']
    print(f'Positive Weight for BCE: {args.bce_pos_weight:.4f}')

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l2_loss_func = nn.MSELoss()
    bce_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.bce_pos_weight).to(device))

    # create data loaders
    print('Creating data loaders...')
    data_start = time.time()
    train_data = AgentTransferable(
        data_home_dirs=args.data_home_dirs, task_names=task_names, split='train',
        iou_thresh=args.iou_thresh, time_interval=args.time_interval, stage=args.stage,
        depth_descriptor=args.depth_descriptor, depth_norm_params=args.depth_norm_params,
        ori_norm_params=args.ori_norm_params, debug=args.debug, run_on_cv_server=args.run_on_cv_server,
        num_cpus=args.num_workers, has_task_labels=True, has_future_labels=True
    )
    data_end = time.time()
    print(f'Loaded train data. Time: {data_end - data_start:.5f} seconds')
    if args.sanity_check:
        print(f'Performing sanity check on {args.sanity_check_size} examples.')
        indices = np.random.choice(range(1, len(train_data)), size=args.sanity_check_size)
        train_data = torch.utils.data.Subset(train_data, indices)
    print(f'There are {len(train_data)} train data.')
    if args.eval_on_train:
        print('Evaluating on training set instead of validation set.')
        valid_data = train_data
    else:
        data_start = time.time()
        valid_data = AgentTransferable(
            data_home_dirs=args.data_home_dirs, task_names=task_names, split='valid',
            iou_thresh=args.iou_thresh, time_interval=args.time_interval, stage=args.stage,
            depth_descriptor=args.depth_descriptor, depth_norm_params=args.depth_norm_params,
            ori_norm_params=args.ori_norm_params, debug=args.debug, run_on_cv_server=args.run_on_cv_server,
            num_cpus=args.num_workers, has_task_labels=True, has_future_labels=True
        )
        data_end = time.time()
        print(f'Loaded valid data. Time: {data_end - data_start:.5f} seconds')
    print(f'There are {len(valid_data)} valid data.')

    dataloader_num_workers = args.num_workers
    if args.debug or args.sanity_check:
        dataloader_num_workers = 0
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.no_shuffle,
        num_workers=dataloader_num_workers, drop_last=True
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=not args.no_shuffle,
        num_workers=dataloader_num_workers, drop_last=True
    )
    print('Creating data loaders: done')

    if args.cont_training:
        if not [f for f in os.listdir(args.save) if 'checkpoint' in f]:
            print(f'No checkpoint found in directory: {args.save}')
            print('Begin training from scratch...')
            global_step, init_epoch = 0, 1
        else:
            checkpoint_name = sorted([f for f in os.listdir(args.save) if 'checkpoint' in f])[-1]
            checkpoint_file = os.path.join(args.save, checkpoint_name)
            print(f'loading checkpoint: {checkpoint_file}')
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            init_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['global_step']
    else:
        print('Begin training from scratch...')
        global_step, init_epoch = 0, 1

    for epoch in range(init_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f'\nepoch {epoch}')

        # Training.
        TrainStats = train(
            train_queue, model, optimizer, device,
            l2_loss_func, bce_loss_func,
            global_step, writer,
            visualizer, hand_mocap,
            task_names, args
        )
        global_step = TrainStats.global_step
        log_epoch_stats(TrainStats, writer, args, global_step, epoch, train=True)

        # Evaluation.
        if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            ValidStats = test(
                valid_queue, model, device,
                l2_loss_func, bce_loss_func,
                global_step, writer,
                visualizer, hand_mocap,
                task_names, args,
            )
            log_epoch_stats(ValidStats, writer, args, global_step, epoch, train=False)

        # Save model.
        if epoch % args.save_freq == 0 or epoch == (args.epochs - 1):
            print('saving the model.')
            torch.save(
                {'epoch': epoch, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'global_step': global_step,
                 'args': args},
                join(args.save, f'checkpoint_{epoch:04d}.pt')
            )

        epoch_end = time.time()
        epoch_elapsed = str(timedelta(seconds=round(epoch_end - epoch_start, 2))).split('.')[0]
        program_elapsed = str(timedelta(seconds=round(epoch_end - program_start, 2))).split('.')[0]
        print(f'Epoch elapsed time: {epoch_elapsed}')
        print(f'Program elapsed time: {program_elapsed}')

    # Final validation.
    print('\nFinal validation.')
    valid_stats = test(
        valid_queue, model, device,
        l2_loss_func, bce_loss_func,
        global_step, writer,
        visualizer, hand_mocap,
        task_names, args,
    )

    log_epoch_stats(valid_stats, writer, args, global_step, args.epochs, train=False)

    program_end = time.time()
    program_elapsed = str(timedelta(seconds=round(program_end - program_start, 2))).split('.')[0]
    print(f'\nDone. Program elapsed time: {program_elapsed}')


def train(
        train_queue, model, optimizer, device,
        l2_loss_func, bce_loss_func,
        global_step, writer,
        visualizer, hand_mocap,
        task_names, args
):
    model.train()

    TrainMonitor = EpochLossMonitor()
    BLcurMonitor = EpochLossMonitor()  # using current hand info for prediction as baseline
    BLmeanMonitor = EpochLossMonitor()  # using mean of batch for prediction as baseline
    EpochMetric = {task_name: {'total': 0, 'pred_success': 0, 'gt_success': 0} for task_name in task_names}
    ContactStats = {c: {'gt': 0, 'pred': 0, 'bl_cur': 0, 'bl_mean': 0} for c in ['pos', 'neg']}

    t0 = time.time()
    for step, data in tqdm(enumerate(train_queue), desc='Going through train data...'):
        if step == 0:
            t1 = time.time()
            print(f'\nDataloader iterator init time: {t1 - t0:.4f}s')

        # process batch data
        input = torch.cat((
            data.hand_r3m, data.task,
            data.current_x.unsqueeze(1), data.current_y.unsqueeze(1), data.current_depth.unsqueeze(1),
            data.current_ori, data.current_contact.unsqueeze(1),
        ), dim=1).to(device).float()

        current_x, current_y = data.current_x.to(device).float(), data.current_y.to(device).float()
        current_xy = torch.cat((current_x.unsqueeze(1), current_y.unsqueeze(1)), dim=1)
        current_depth, current_ori = data.current_depth.to(device).float(), data.current_ori.to(device)
        current_contact = data.current_contact.to(device).float()

        future_x, future_y = data.future_x.to(device).float(), data.future_y.to(device).float()
        future_xy = torch.cat((future_x.unsqueeze(1), future_y.unsqueeze(1)), dim=1)
        future_depth, future_ori = data.future_depth.to(device).float(), data.future_ori.to(device)
        future_contact = data.future_contact.to(device).float()

        # handle residual prediction mode
        target_xy = get_target_val(future_xy, current_xy, args)
        target_depth = get_target_val(future_depth, current_depth, args)
        target_ori = get_target_val(future_ori, current_ori, args)

        # forward through network
        optimizer.zero_grad()
        output = model(input)

        # process network output
        pred_xy, pred_depth = output[:, 0:2], output[:, 2]
        pred_ori, pred_contact = output[:, 3:6], output[:, 6]
        pred_contact_binary = torch.round(torch.sigmoid(pred_contact))  # float -> 0/1
        if not args.pred_residual:
            pred_xy = torch.sigmoid(pred_xy)  # force xy to be positive bbox coords

        # process contact and loss
        xy_loss = l2_loss_func(pred_xy, target_xy)
        depth_loss = l2_loss_func(pred_depth, target_depth)
        ori_loss = l2_loss_func(pred_ori, target_ori)
        contact_loss = bce_loss_func(pred_contact, future_contact)
        loss = args.lambda1 * xy_loss + \
               args.lambda2 * depth_loss + \
               args.lambda3 * ori_loss + \
               args.lambda4 * contact_loss

        with torch.no_grad():
            contact_accs, ContactStats = process_contact_stats(
                current_contact, future_contact, pred_contact, ContactStats
            )
            contact_acc, cur_contact_acc, mean_contact_acc = contact_accs

            loss_unweighted = xy_loss + depth_loss + ori_loss + contact_loss
            TrainLosses = LossStats(
                total_loss=loss_unweighted.data, xy_loss=xy_loss.data, depth_loss=depth_loss.data,
                ori_loss=ori_loss.data, contact_loss=contact_loss.data, contact_acc=contact_acc.data,
            )
            BLcurLosses = process_baseline_loss(
                current_data=(current_xy, current_depth, current_ori, current_contact),
                target_data=(target_xy, target_depth, target_ori, future_contact),
                contact_accs=(cur_contact_acc, mean_contact_acc),
                loss_funcs=(l2_loss_func, bce_loss_func), args=args, mode='current'
            )
            BLmeanLosses = process_baseline_loss(
                current_data=(current_xy, current_depth, current_ori, current_contact),
                target_data=(target_xy, target_depth, target_ori, future_contact),
                contact_accs=(cur_contact_acc, mean_contact_acc),
                loss_funcs=(l2_loss_func, bce_loss_func), args=args, mode='mean'
            )

        # back propogate
        loss.backward()
        optimizer.step()

        # process metric evaluation
        metric_stats = evaluate_transferable_metric_batch(
            task_names=task_names, task=data.task, device=device,
            current_x=current_x, future_x=future_x, current_y=current_y, future_y=future_y,
            current_depth=current_depth, future_depth=future_depth, evaluate_gt=True,
            pred_x=get_sum_val(current_x, pred_xy[:, 0], args),
            pred_y=get_sum_val(current_y, pred_xy[:, 1], args),
            pred_depth=get_sum_val(current_depth, pred_depth, args)
        )
        for k, v in metric_stats.items():
            EpochMetric[k]['total'] += v['total']
            EpochMetric[k]['pred_success'] += v['pred_success']
            EpochMetric[k]['gt_success'] += v['gt_success']

        TrainMonitor.update(TrainLosses, 1)
        BLcurMonitor.update(BLcurLosses, 1)
        BLmeanMonitor.update(BLmeanLosses, 1)

        # log scalars
        if (global_step + 1) % args.log_scalar_freq == 0:
            write_loss_to_tb(TrainLosses, writer, global_step, tag='train')
            write_loss_to_tb(BLcurLosses, writer, global_step, tag='train_baseline_current')
            write_loss_to_tb(BLmeanLosses, writer, global_step, tag='train_baseline_mean')

        # log images
        if (global_step + 1) % args.vis_freq == 0 and not args.sanity_check:
            # visualize some samples in the batch
            vis_imgs = []
            for i in range(args.vis_sample_size):
                task_name = task_names[torch.argmax(data.task[i].squeeze())]
                passed_metric = evaluate_transferable_metric(
                    task_name=task_name, current_x=current_x[i], current_y=current_y[i], current_depth=current_depth[i],
                    pred_x=get_sum_val(current_x[i], pred_xy[i, 0], args),
                    pred_y=get_sum_val(current_y[i], pred_xy[i, 1], args),
                    pred_depth=get_sum_val(current_depth[i], pred_depth[i], args)
                )

                vis_img = generate_transferable_visualization(
                    current_hand_pose_path=data.current_info_path[i], future_hand_pose_path=data.future_info_path[i],
                    run_on_cv_server=args.run_on_cv_server, hand=data.hand[i],
                    pred_delta_x=get_res_val(pred_xy[i, 0], current_x[i], args).item(),
                    pred_delta_y=get_res_val(pred_xy[i, 1], current_y[i], args).item(),
                    pred_delta_depth=get_res_val(pred_depth[i], current_depth[i], args).item(),
                    pred_delta_ori=get_res_val(pred_ori[i], current_ori[i], args),
                    pred_contact=pred_contact_binary[i],
                    depth_norm_params=args.depth_norm_params, ori_norm_params=args.ori_norm_params,
                    task_name=task_name, visualizer=visualizer, use_visualizer=args.use_visualizer,
                    hand_mocap=hand_mocap, device=device, log_metric=True, passed_metric=passed_metric.item()
                )
                vis_imgs.append(vis_img)
            final_vis_img = np.hstack(vis_imgs)
            writer.add_image(f'train/vis_images', final_vis_img, global_step, dataformats='HWC')

            # visualize different task conditioning
            if args.eval_tasks:
                for i in range(args.task_vis_sample_size):
                    # process task input
                    original_task_name = task_names[torch.argmax(data.task[i].squeeze())]
                    all_task_instances = []
                    for j, task_name in enumerate(task_names):
                        task_instance = torch.zeros(1, len(task_names))
                        task_instance[0, j] = 1
                        all_task_instances.append(task_instance)
                    all_task_instances = torch.vstack(all_task_instances)

                    task_conditioned_input = torch.cat((
                        data.hand_r3m[i].repeat(len(task_names), 1), all_task_instances,
                        data.current_x[i].repeat(len(task_names), 1), data.current_y[i].repeat(len(task_names), 1),
                        data.current_depth[i].repeat(len(task_names), 1),
                        data.current_ori[i].repeat(len(task_names), 1),
                        data.current_contact[i].repeat(len(task_names), 1),
                    ), dim=1).to(device).float()

                    # forward through network and process output
                    model.eval()
                    with torch.no_grad():
                        task_conditioned_output = model(task_conditioned_input)
                        t_pred_xy, t_pred_depth = task_conditioned_output[:, 0:2], task_conditioned_output[:, 2]
                        t_pred_ori, t_pred_contact = task_conditioned_output[:, 3:6], task_conditioned_output[:, 6]
                        if not args.pred_residual:
                            t_pred_xy = torch.sigmoid(t_pred_xy)  # force xy to be positive bbox coords

                    task_vis_imgs = []
                    for j, task_name in enumerate(task_names):
                        passed_metric = evaluate_transferable_metric(
                            task_name=task_name,
                            current_x=current_x[i], current_y=current_y[i], current_depth=current_depth[i],
                            pred_x=get_sum_val(current_x[i], t_pred_xy[j, 0], args),
                            pred_y=get_sum_val(current_y[i], t_pred_xy[j, 1], args),
                            pred_depth=get_sum_val(current_depth[i], t_pred_depth[j], args)
                        )

                        pred_contact_binary = torch.round(torch.sigmoid(t_pred_contact[j]))
                        task_vis_img = generate_transferable_visualization(
                            current_hand_pose_path=data.current_info_path[i], future_hand_pose_path=None,
                            run_on_cv_server=args.run_on_cv_server, hand=data.hand[i],
                            pred_delta_x=get_res_val(t_pred_xy[j, 0], current_x[i], args).item(),
                            pred_delta_y=get_res_val(t_pred_xy[j, 1], current_y[i], args).item(),
                            pred_delta_depth=get_res_val(t_pred_depth[j], current_depth[i], args).item(),
                            pred_delta_ori=get_res_val(t_pred_ori[j], current_ori[i], args),
                            pred_contact=pred_contact_binary,
                            depth_norm_params=args.depth_norm_params, ori_norm_params=args.ori_norm_params,
                            task_name=task_name, visualizer=visualizer, use_visualizer=args.use_visualizer,
                            hand_mocap=hand_mocap, device=device, log_metric=True, passed_metric=passed_metric.item(),
                            original_task=task_name == original_task_name, vis_groundtruth=False
                        )
                        task_vis_imgs.append(task_vis_img)
                    final_task_vis_img = np.hstack(task_vis_imgs)
                    writer.add_image(
                        f'train_eval_tasks/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC'
                    )

            model.train()

        global_step += 1

    return EpochStats(
        LossMonitor=TrainMonitor, BLcurMonitor=BLcurMonitor, BLmeanMonitor=BLmeanMonitor,
        EpochMetric=EpochMetric, TaskMetric=None, ContactStats=ContactStats, global_step=global_step
    )


def test(
        valid_queue, model, device,
        l2_loss_func, bce_loss_func,
        global_step, writer,
        visualizer, hand_mocap,
        task_names, args
):
    model.eval()

    ValidMonitor = EpochLossMonitor()
    BLcurMonitor = EpochLossMonitor()  # using current hand info for prediction as baseline
    BLmeanMonitor = EpochLossMonitor()  # using mean of batch for prediction as baseline
    EpochMetric = {task_name: {'total': 0, 'pred_success': 0, 'gt_success': 0} for task_name in task_names}
    TaskMetric = None
    ContactStats = {c: {'gt': 0, 'pred': 0, 'bl_cur': 0, 'bl_mean': 0} for c in ['pos', 'neg']}

    data, current_x, current_y, current_depth, current_ori = None, None, None, None, None
    for step, data in tqdm(enumerate(valid_queue), 'Going through valid data...'):
        # process batch data
        input = torch.cat((
            data.hand_r3m, data.task,
            data.current_x.unsqueeze(1), data.current_y.unsqueeze(1), data.current_depth.unsqueeze(1),
            data.current_ori, data.current_contact.unsqueeze(1),
        ), dim=1).to(device).float()

        current_x, current_y = data.current_x.to(device).float(), data.current_y.to(device).float()
        current_xy = torch.cat((current_x.unsqueeze(1), current_y.unsqueeze(1)), dim=1)
        current_depth, current_ori = data.current_depth.to(device).float(), data.current_ori.to(device)
        current_contact = data.current_contact.to(device).float()

        future_x, future_y = data.future_x.to(device).float(), data.future_y.to(device).float()
        future_xy = torch.cat((future_x.unsqueeze(1), future_y.unsqueeze(1)), dim=1)
        future_depth, future_ori = data.future_depth.to(device).float(), data.future_ori.to(device)
        future_contact = data.future_contact.to(device).float()

        # handle residual prediction mode
        target_xy = get_target_val(future_xy, current_xy, args)
        target_depth = get_target_val(future_depth, current_depth, args)
        target_ori = get_target_val(future_ori, current_ori, args)

        batch_size = target_xy.size(0)
        with torch.no_grad():
            # forward through model and process output
            output = model(input)
            pred_xy, pred_depth = output[:, 0:2], output[:, 2]
            pred_ori, pred_contact = output[:, 3:6], output[:, 6]
            pred_contact_binary = torch.round(torch.sigmoid(pred_contact))  # float -> 0/1
            if not args.pred_residual:
                pred_xy = torch.sigmoid(pred_xy)  # force xy to be positive bbox coords

            # process contact and loss
            xy_loss = l2_loss_func(pred_xy, target_xy)
            depth_loss = l2_loss_func(pred_depth, target_depth)
            ori_loss = l2_loss_func(pred_ori, target_ori)
            contact_loss = bce_loss_func(pred_contact, future_contact)
            loss = args.lambda1 * xy_loss + \
                   args.lambda2 * depth_loss + \
                   args.lambda3 * ori_loss + \
                   args.lambda4 * contact_loss

            contact_accs, ContactStats = process_contact_stats(
                current_contact, future_contact, pred_contact, ContactStats
            )
            contact_acc, cur_contact_acc, mean_contact_acc = contact_accs

            loss_unweighted = xy_loss + depth_loss + ori_loss + contact_loss
            ValidLosses = LossStats(
                total_loss=loss_unweighted.data, xy_loss=xy_loss.data, depth_loss=depth_loss.data,
                ori_loss=ori_loss.data, contact_loss=contact_loss.data, contact_acc=contact_acc.data,
            )

            # process baseline
            BLcurLosses = process_baseline_loss(
                current_data=(current_xy, current_depth, current_ori, current_contact),
                target_data=(target_xy, target_depth, target_ori, future_contact),
                contact_accs=(cur_contact_acc, mean_contact_acc),
                loss_funcs=(l2_loss_func, bce_loss_func), args=args, mode='current'
            )
            BLmeanLosses = process_baseline_loss(
                current_data=(current_xy, current_depth, current_ori, current_contact),
                target_data=(target_xy, target_depth, target_ori, future_contact),
                contact_accs=(cur_contact_acc, mean_contact_acc),
                loss_funcs=(l2_loss_func, bce_loss_func), args=args, mode='mean'
            )

            # process metric evaluation
            metric_stats = evaluate_transferable_metric_batch(
                task_names=task_names, task=data.task, device=device,
                current_x=current_x, future_x=future_x, current_y=current_y, future_y=future_y,
                current_depth=current_depth, future_depth=future_depth, evaluate_gt=True,
                pred_x=get_sum_val(current_x, pred_xy[:, 0], args),
                pred_y=get_sum_val(current_y, pred_xy[:, 1], args),
                pred_depth=get_sum_val(current_depth, pred_depth, args)
            )
            for k, v in metric_stats.items():
                EpochMetric[k]['total'] += v['total']
                EpochMetric[k]['pred_success'] += v['pred_success']
                EpochMetric[k]['gt_success'] += v['gt_success']

        ValidMonitor.update(ValidLosses, batch_size)
        BLcurMonitor.update(BLcurLosses, batch_size)
        BLmeanMonitor.update(BLmeanLosses, batch_size)

    # visualize some samples in the last batch
    vis_imgs = []
    for i in range(args.vis_sample_size):
        task_name = task_names[torch.argmax(data.task[i].squeeze())]
        passed_metric = evaluate_transferable_metric(
            task_name=task_name,
            current_x=current_x[i], current_y=current_y[i], current_depth=current_depth[i],
            pred_x=get_sum_val(current_x[i], pred_xy[i, 0], args),
            pred_y=get_sum_val(current_y[i], pred_xy[i, 1], args),
            pred_depth=get_sum_val(current_depth[i], pred_depth[i], args)
        )

        vis_img = generate_transferable_visualization(
            current_hand_pose_path=data.current_info_path[i], future_hand_pose_path=data.future_info_path[i],
            run_on_cv_server=args.run_on_cv_server, hand=data.hand[i],
            pred_delta_x=get_res_val(pred_xy[i, 0], current_x[i], args).item(),
            pred_delta_y=get_res_val(pred_xy[i, 1], current_y[i], args).item(),
            pred_delta_depth=get_res_val(pred_depth[i], current_depth[i], args).item(),
            pred_delta_ori=get_res_val(pred_ori[i], current_ori[i], args),
            pred_contact=pred_contact_binary[i],
            depth_norm_params=args.depth_norm_params, ori_norm_params=args.ori_norm_params,
            task_name=task_name, visualizer=visualizer, use_visualizer=args.use_visualizer,
            hand_mocap=hand_mocap, device=device, log_metric=True, passed_metric=passed_metric.item()
        )
        vis_imgs.append(vis_img)
    final_vis_img = np.hstack(vis_imgs)
    writer.add_image(f'valid/vis_images', final_vis_img, global_step, dataformats='HWC')

    # task-conditioned evaluation
    if args.eval_tasks:
        task_vis_sample_count = 0
        TaskMetric = {task_name: {'total': 0, 'pred_success': 0} for task_name in task_names}
        all_task_instances = []
        for j, task_name in enumerate(task_names):
            task_instance = torch.zeros(1, len(task_names))
            task_instance[0, j] = 1
            all_task_instances.append(task_instance)
        all_task_instances = torch.vstack(all_task_instances)
        for i in range(data.current_x.size(0)):
            original_task_name = task_names[torch.argmax(data.task[i].squeeze())]
            task_conditioned_input = torch.cat((
                data.hand_r3m[i].repeat(len(task_names), 1), all_task_instances,
                data.current_x[i].repeat(len(task_names), 1), data.current_y[i].repeat(len(task_names), 1),
                data.current_depth[i].repeat(len(task_names), 1), data.current_ori[i].repeat(len(task_names), 1),
                data.current_contact[i].repeat(len(task_names), 1),
            ), dim=1).to(device).float()

            with torch.no_grad():
                task_conditioned_output = model(task_conditioned_input)
                t_pred_xy, t_pred_depth = task_conditioned_output[:, 0:2], task_conditioned_output[:, 2]
                t_pred_ori, t_pred_contact = task_conditioned_output[:, 3:6], task_conditioned_output[:, 6]
                if not args.pred_residual:
                    t_pred_xy = torch.sigmoid(t_pred_xy)  # force xy to be positive bbox coords

            # process metric evaluation
            t_current_x = data.current_x[i].repeat(len(task_names)).to(device).float()
            t_current_y = data.current_y[i].repeat(len(task_names)).to(device).float()
            t_current_depth = data.current_depth[i].repeat(len(task_names)).to(device).float()

            metric_stats = evaluate_transferable_metric_batch(
                task_names=task_names, task=all_task_instances, device=device, evaluate_gt=False,
                current_x=t_current_x, current_y=t_current_y, current_depth=t_current_depth,
                pred_x=get_sum_val(current_x, t_pred_xy[:, 0], args),
                pred_y=get_sum_val(current_y, t_pred_xy[:, 1], args),
                pred_depth=get_sum_val(current_depth, t_pred_depth, args)
            )
            for k, v in metric_stats.items():
                TaskMetric[k]['total'] += v['total']
                TaskMetric[k]['pred_success'] += v['pred_success']

            # visualize some samples of task-conditioned evaluation
            if task_vis_sample_count < args.task_vis_sample_size:
                task_vis_imgs = []
                for j, task_name in enumerate(task_names):
                    pred_contact_binary = torch.round(torch.sigmoid(t_pred_contact[j]))
                    passed_metric = evaluate_transferable_metric(
                        task_name=task_name,
                        current_x=current_x[i], current_y=current_y[i], current_depth=current_depth[i],
                        pred_x=get_sum_val(current_x[i], t_pred_xy[j, 0], args),
                        pred_y=get_sum_val(current_y[i], t_pred_xy[j, 1], args),
                        pred_depth=get_sum_val(current_depth[i], t_pred_depth[j], args)
                    )

                    task_vis_img = generate_transferable_visualization(
                        current_hand_pose_path=data.current_info_path[i], future_hand_pose_path=None,
                        run_on_cv_server=args.run_on_cv_server, hand=data.hand[i],
                        pred_delta_x=get_res_val(t_pred_xy[j, 0], current_x[i], args).item(),
                        pred_delta_y=get_res_val(t_pred_xy[j, 1], current_y[i], args).item(),
                        pred_delta_depth=get_res_val(t_pred_depth[j], current_depth[i], args).item(),
                        pred_delta_ori=get_res_val(t_pred_ori[j], current_ori[i], args),
                        pred_contact=pred_contact_binary,
                        depth_norm_params=args.depth_norm_params, ori_norm_params=args.ori_norm_params,
                        task_name=task_name, visualizer=visualizer, use_visualizer=args.use_visualizer,
                        hand_mocap=hand_mocap, device=device, log_metric=True, passed_metric=passed_metric.item(),
                        original_task=task_name == original_task_name, vis_groundtruth=False
                    )
                    task_vis_imgs.append(task_vis_img)
                final_task_vis_img = np.hstack(task_vis_imgs)
                writer.add_image(f'valid_eval_tasks/vis_tasks_{i}', final_task_vis_img, global_step, dataformats='HWC')
                task_vis_sample_count += 1

    return EpochStats(
        LossMonitor=ValidMonitor, BLcurMonitor=BLcurMonitor, BLmeanMonitor=BLmeanMonitor,
        EpochMetric=EpochMetric, TaskMetric=TaskMetric, ContactStats=ContactStats, global_step=global_step
    )


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    main(args)
