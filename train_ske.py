import os
import torch
import numpy as np
import argparse
import datetime

from pathlib import Path
from tqdm import tqdm
from utils import provider
from utils.logger import Logger
from utils.data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.SkelPointEdgeNet import SkelPointEdgeNet

# Train skeleton network
# CUDA_VISIBLE_DEVICES=0 python train_ske.py --use_delaunay --pre_train_epoch 20 --ske_train_epoch 60 --ske_batch_size 4 --num_skeleton_point 128


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_delaunay', type=int, default=32, help='Delaunay')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_true', default=False, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    # skeleton network
    parser.add_argument('--optimizer_ske', type=str, default='Adam', help='optimizer for skeleton training')
    parser.add_argument('--lr_ske', type=float, default=0.001, help='')
    parser.add_argument('--pre_train_epoch', type=int, default=20, help='')
    parser.add_argument('--ske_train_epoch', type=int, default=60, help='')
    parser.add_argument('--ske_batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--use_pretrained_skeleton', action='store_true', default=False, help='use pretrain model')
    parser.add_argument('--pretrain_skeleton', type=str, default='', help='')
    parser.add_argument('--num_skeleton_point', type=int, default=128, help='Skeleton Point Number')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')

    parser.add_argument('--save_ske_net_path', type=str, default='checkpoints_skeletons', help='directory to save the network parameters')
    parser.add_argument('--save_ske_train_path', type=str, default='train_skeletons', help='directory to save the temporary results during training')
    parser.add_argument('--save_ske_test_path', type=str, default='test_skeletons', help='directory to save the testing results')
    parser.add_argument('--save_ske_iter', type=int, default=1000, help='frequency to save the intermediate results (number of iteration)')
    parser.add_argument('--save_ske_epoch', type=int, default=5, help='frequency to save the network parameters (number of epoch)')

    return parser.parse_args()


def output_results(log_path, batch_id, epoch, input_xyz, ske_xyz, ske_r):
    batch_size = ske_xyz.size()[0]

    input_xyz_save = input_xyz.detach().cpu().numpy()
    ske_xyz_save = ske_xyz.detach().cpu().numpy()
    ske_r_save = ske_r.detach().cpu().numpy()

    for i in range(batch_size):
        provider.save_off_points(input_xyz_save[i], os.path.join(log_path, str(batch_id[i]) + "_input.off"))
        provider.save_spheres(ske_xyz_save[i], ske_r_save[i], os.path.join(log_path, str(batch_id[i]) + "_sphere_" + str(epoch) + ".obj"))
        provider.save_off_points(ske_xyz_save[i], os.path.join(log_path, str(batch_id[i]) + "_center_" + str(epoch) + ".off"))


def main():
    args = parse_args()

    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath('skeleton_' + timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir + timestr)
    exp_dir.mkdir(exist_ok=True)

    '''SKELETON DIR'''
    ske_checkpoint_dir = exp_dir.joinpath(args.save_ske_net_path)
    ske_checkpoint_dir.mkdir(exist_ok=True)
    ske_train_dir = exp_dir.joinpath(args.save_ske_train_path)
    ske_train_dir.mkdir(exist_ok=True)
    ske_test_dir = exp_dir.joinpath(args.save_ske_test_path)
    ske_test_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = Logger('%s/%s.txt' % (exp_dir, 'sampling_' + timestr))
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    # ------------------ ModelNet -----------------
    data_path = 'data/modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')

    # Skeleton dataloader
    train_loader_ske = torch.utils.data.DataLoader(train_dataset, batch_size=args.ske_batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader_ske = torch.utils.data.DataLoader(test_dataset, batch_size=args.ske_batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # -------------------- Skeleton network --------------------
    # pointnet++ backbone
    # model_skel = SkelPointNet(num_skel_points=args.num_skeleton_point, input_channels=0, use_xyz=True).to(device)
    # dgcnn backbone
    model_ske = SkelPointEdgeNet(args).to(device)
    if args.optimizer_ske == 'Adam':
        optimizer_ske = torch.optim.Adam(model_ske.parameters(), lr=args.lr_ske)
    else:
        optimizer_ske = torch.optim.SGD(model_ske.parameters(), lr=0.01, momentum=0.9)

    if args.use_pretrained_skeleton:
        try:
            checkpoint = torch.load(args.pretrain_skeleton)
            model_ske.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Use pretrained skeleton')
        except:
            logger.info('Error: No existing skeleton')
            exit()

    '''TRANING'''
    logger.info('Start training ...')

    global_epoch, global_step = 0, 0
    start_ske_epoch, ske_epoch = 0, args.pre_train_epoch + args.ske_train_epoch

    best_test_loss = np.inf

    for epoch in range(start_ske_epoch, ske_epoch):
        logger.info('Skeleton Epoch {:0>{}}/{}:'.format(epoch + 1, len(str(ske_epoch)), ske_epoch))

        model_ske.train()

        total_loss = 0.

        starttime = datetime.datetime.now()

        tbar = tqdm(enumerate(train_loader_ske, 0), total=len(train_loader_ske), smoothing=0.9)
        for batch_id, (points, _, pnames, dly_idx) in tbar:
            points = points.float().to(device)
            dly_idx = dly_idx.long().to(device)

            if epoch < args.pre_train_epoch:
                # Pre-train skeletal point network
                tbar.set_description("Pre-training", refresh=True)

                ske_xyz, ske_r, shape_features = model_ske(points, dly_idx)
                loss_pre = model_ske.compute_loss_pre(points, ske_xyz)

                optimizer_ske.zero_grad()
                loss_pre.backward()
                optimizer_ske.step()

                total_loss += loss_pre
            else:
                # Train skeletal point network with geometric losses
                tbar.set_description("Skeletal Point Training", refresh=True)

                ske_xyz, ske_r, shape_features = model_ske(points, dly_idx)
                loss_ske = model_ske.compute_loss(points, ske_xyz, ske_r, None, 0.3, 0.4)

                optimizer_ske.zero_grad()
                loss_ske.backward()
                optimizer_ske.step()

                total_loss += loss_ske

                # Random save results during training
                if (global_step + 1) % args.save_ske_iter == 0:
                    output_results(str(ske_train_dir), pnames, epoch, points, ske_xyz, ske_r)

            global_step += 1

        train_loss = total_loss / float(batch_id + 1)
        logger.info('Train Loss: %f' % train_loss)

        if not epoch < args.pre_train_epoch:
            if (epoch + 1) % args.save_ske_epoch == 0:
                state = {'epoch': epoch + 1, 'model_state_dict': model_ske.state_dict(), 'optimizer_state_dict': optimizer_ske.state_dict()}
                savepath = os.path.join(str(ske_checkpoint_dir), 'skeleton_%04depoch.pth' % (epoch + 1))
                torch.save(state, savepath)

            with torch.no_grad():
                model_ske.eval()
                total_loss = 0.

                for batch_id, (points, _, pnames, dly_idx) in tqdm(enumerate(test_loader_ske), total=len(test_loader_ske)):
                    points = points.float().to(device)
                    dly_idx = dly_idx.long().to(device)

                    ske_xyz, ske_r, shape_features = model_ske(points, dly_idx)
                    loss_ske = model_ske.compute_loss(points, ske_xyz, ske_r, None, 0.3, 0.4)
                    total_loss += loss_ske

                    if (epoch + 1) == ske_epoch:
                        ske_test_epoch_dir = ske_test_dir.joinpath(str(epoch + 1))
                        ske_test_epoch_dir.mkdir(exist_ok=True)
                        output_results(str(ske_test_epoch_dir), pnames, 'testing', points, ske_xyz, ske_r)

                test_loss = total_loss / float(batch_id + 1)
                if best_test_loss >= test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch + 1

                    state = {'epoch': best_epoch, 'model_state_dict': model_ske.state_dict(), 'optimizer_state_dict': optimizer_ske.state_dict()}
                    savepath = os.path.join(str(ske_checkpoint_dir), 'skeleton_best.pth')
                    torch.save(state, savepath)
                    logger.info('Saving at %s' % savepath)

                logger.info('Test Loss: %f' % test_loss)
                logger.info('Best Test Loss: %f' % best_test_loss)

        global_epoch += 1
        logger.info('Epoch time %f(m)' % ((datetime.datetime.now() - starttime).total_seconds() / 60.0))
    logger.info('End of skeleton training ...')


if __name__ == '__main__':
    main()
