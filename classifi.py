import os
import sys
import torch
import numpy as np
import argparse
import datetime

from pathlib import Path
from tqdm import tqdm
from utils import provider
from utils import mesh_utils
from utils.logger import Logger
from utils.data_utils.ModelNetDataLoader import ModelNetDataLoader, ModelNetTestDataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classification'))

from models.Sampling import InfoSampling
from models.SkelPointEdgeNet import SkelPointEdgeNet
from classification.models.pointnet_cls import PointNet

# Train sampling network
# CUDA_VISIBLE_DEVICES=0 python classifi.py --use_delaunay --spl_epoch 250 --spl_batch_size 4 --num_sampling 128 --use_pretrained_skeleton --pretrain_skeleton ... --num_skeleton_point ...

# Train sampling network
# CUDA_VISIBLE_DEVICES=0 python classifi.py --test


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--test', action='store_true', default=False, help='training or testing')

    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_delaunay', type=int, default=32, help='Delaunay')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_true', default=False, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    # sampling network
    parser.add_argument('--optimizer_spl', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr_spl', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--spl_epoch', default=250, type=int, help='number of epoch in training')
    parser.add_argument('--spl_batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--use_pretrained_skeleton', action='store_true', default=False, help='use pretrain model')
    parser.add_argument('--pretrain_skeleton', type=str, default='', help='')
    parser.add_argument('--num_skeleton_point', type=int, default=128, help='Skeleton Point Number')
    parser.add_argument('--num_sampling', type=int, default=128, metavar='N', help='Num of sampling points')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--bins', type=int, default=6, metavar='N', help='Num of histogram bins')

    parser.add_argument('--save_spl_net_path', type=str, default='checkpoints_sampling', help='directory to save the network parameters')
    parser.add_argument('--save_spl_train_path', type=str, default='train_sampling', help='directory to save the temporary results during training')
    parser.add_argument('--save_spl_test_path', type=str, default='test_sampling', help='directory to save the testing results')
    parser.add_argument('--save_spl_epoch', type=int, default=25, help='frequency to save the network parameters (number of epoch)')

    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath('sampling_' + timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir + timestr)
    exp_dir.mkdir(exist_ok=True)

    '''SAMPLING DIR'''
    spl_checkpoint_dir = exp_dir.joinpath(args.save_spl_net_path)
    spl_checkpoint_dir.mkdir(exist_ok=True)
    spl_train_dir = exp_dir.joinpath(args.save_spl_train_path)
    spl_train_dir.mkdir(exist_ok=True)
    spl_test_dir = exp_dir.joinpath(args.save_spl_test_path)
    spl_test_dir.mkdir(exist_ok=True)

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

    # Sampling dataloader
    train_loader_spl = torch.utils.data.DataLoader(train_dataset, batch_size=args.spl_batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader_spl = torch.utils.data.DataLoader(test_dataset, batch_size=args.spl_batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # -------------------- Skeleton network --------------------
    # pointnet++ backbone
    # model_skel = SkelPointNet(num_skel_points=args.num_skeleton_point, input_channels=0, use_xyz=True).to(device)
    # dgcnn backbone
    model_ske = SkelPointEdgeNet(args).to(device)

    if args.use_pretrained_skeleton:
        try:
            checkpoint = torch.load(args.pretrain_skeleton)
            model_ske.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Use pretrained skeleton')
        except:
            logger.info('Error: No existing skeleton')
            exit()

    # -------------------- Sampling network --------------------
    model_spl = InfoSampling(args).to(device)
    if args.optimizer_spl == 'Adam':
        optimizer_spl = torch.optim.Adam(model_spl.parameters(), lr=args.lr_spl)
    else:
        optimizer_spl = torch.optim.SGD(model_spl.parameters(), lr=0.01, momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # -------------------- Task network --------------------
    # classification
    model_cls = PointNet(args).to(device)
    criterion = model_cls.loss_function

    try:
        checkpoint = torch.load('./classification/log/pointnet_2022-10-24-20-25_seed_390/best_model.pth')
        model_cls.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrained classification model')
        # ------------ test code ------------
        # with torch.no_grad():
        #     model_cls.eval()
        #
        #     total_correct = 0
        #     total_seen = 0
        #     class_acc = np.zeros((args.num_class, 3)).astype(float)
        #
        #     for _, (points, target, _, _) in tqdm(enumerate(testDataLoader_spl), total=len(testDataLoader_spl)):
        #         # if not args.use_cpu:
        #         #     points, target = points.cuda(), target.cuda()
        #
        #         points, target = points.float().to(device), target.long().to(device)
        #         points = points.transpose(2, 1)
        #         pred, _ = model_cls(points)
        #
        #         pred_choice = pred.max(1)[1]
        #         correct = pred_choice.eq(target).sum()
        #         total_correct += correct.item()
        #         total_seen += target.size()[0]
        #
        #         for cat in np.unique(target.cpu()):
        #             classacc = pred_choice[target == cat].eq(target[target == cat]).sum()
        #             class_acc[cat, 0] += classacc.item()
        #             class_acc[cat, 1] += target[target == cat].size()[0]
        #
        #     instance_acc = total_correct / float(total_seen)
        #     class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        #     class_acc = float(np.mean(class_acc[:, 2]))
        #
        #     logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        # exit()
        # ------------ test code ------------
    except:
        logger.info('Error: No existing classification model')
        exit()

    '''TRANING'''
    logger.info('Start training ...')

    ini_temp, min_temp = 1.0, 0.5
    temp = ini_temp

    anneal_rate = -np.log(min_temp / ini_temp) / (int(len(train_dataset) / args.spl_batch_size) * args.spl_epoch * 0.8)
    anneal_interval = 100

    # train_loss_avg = 0.
    # train_loss_interval = 100

    global_epoch, global_step = 0, 0
    start_spl_epoch, spl_epoch = 0, args.spl_epoch

    best_instance_acc = 0.0
    best_class_acc = 0.0

    for epoch in range(start_spl_epoch, spl_epoch):
        logger.info('Sampling Epoch {:0>{}}/{}:'.format(epoch + 1, len(str(spl_epoch)), spl_epoch))

        total_correct = 0
        total_seen = 0
        hard = False

        model_ske.train(mode=False)
        model_spl.train(mode=True)
        model_cls.train(mode=False)

        starttime = datetime.datetime.now()

        for batch_id, (points, target, _, dly_idx) in tqdm(enumerate(train_loader_spl, 0), total=len(train_loader_spl), smoothing=0.9):
            optimizer_spl.zero_grad()

            # Augmentation
            # points = points.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # points = torch.Tensor(points)

            points, target = points.float().to(device), target.long().to(device)
            dly_idx = dly_idx.long().to(device)

            if args.use_pretrained_skeleton:
                # prior sampling weight
                ske_xyz, _, _ = model_ske(points, dly_idx)
                min_dists, _ = mesh_utils.closest_distance_with_batch(points, ske_xyz, is_sum=False)

                hist_edge = provider.linspace_nd(min_dists.min(dim=-1)[0], min_dists.max(dim=-1)[0], steps=args.bins+1)
                hist_edge_mid = torch.mean(torch.stack([hist_edge[:, :-1], hist_edge[:, 1:]]), dim=0)
                hist_edge_weight = torch.exp(-hist_edge_mid) / torch.sum(torch.exp(-hist_edge_mid), dim=-1, keepdim=True)

                prior_weight = []
                for i in range(min_dists.shape[0]):
                    hist_edge_i = hist_edge[i]
                    hist_edge_i[0] -= 0.1
                    hist_index_i = torch.bucketize(min_dists[i], hist_edge_i)
                    hist_index_i -= 1

                    hist_edge_weight_i = hist_edge_weight[i]
                    prior_weight.append(torch.index_select(hist_edge_weight_i, 0, hist_index_i))

                prior_weight = torch.stack(prior_weight, dim=0).detach()
            else:
                # for testing
                prior_weight = torch.ones(points.size(0), points.size(1), dtype=torch.float).to(device).detach()

            points = points.transpose(1, 2)
            sampled_features, attn, stat_joint, stat_margin = model_spl(points, temp, hard, prior_weight)
            sampled_points = torch.matmul(attn, points.transpose(1, 2))

            # sampled_points = sampled_points.transpose(1, 2)
            # mutual information loss
            # expec_joint = (-F.softplus(-stat_joint)).mean()
            # expec_margin = F.softplus(stat_margin).mean()
            # mutual_info = expec_joint - expec_margin
            # loss = -mutual_info

            # classification
            pred = model_cls(sampled_points.transpose(1, 2))
            loss_cls = criterion(pred, target)

            loss = loss_cls

            # optimization
            loss.backward()
            optimizer_spl.step()

            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(target).sum()
            total_correct += correct.item()
            total_seen += target.size(0)

            # train_loss_avg += loss
            # if (batch_id + 1) % train_loss_interval == 0:
            #     logger.info('Batch %d (%d/%s), mean loss: %f' % (batch_id + 1, batch_id + 1, len(trainDataLoader), train_loss_avg / train_loss_interval))
            #     train_loss_avg = 0.

            # annealing
            if (global_step + 1) % anneal_interval == 0:
                temp = np.maximum(ini_temp * np.exp(-anneal_rate * global_step), min_temp)

            global_step += 1

        train_instance_acc = total_correct / float(total_seen)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        if (epoch + 1) % args.save_spl_epoch == 0:
            state = {'epoch': epoch + 1, 'model_state_dict': model_spl.state_dict(), 'optimizer_state_dict': optimizer_spl.state_dict()}
            savepath = os.path.join(str(spl_checkpoint_dir), 'sampling_%04depoch.pth' % (epoch + 1))
            torch.save(state, savepath)

        with torch.no_grad():
            model_ske.eval()
            model_spl.eval()
            model_cls.eval()

            total_correct = 0
            total_seen = 0
            class_acc = np.zeros((args.num_class, 3)).astype(float)

            hard = True
            sampled_points_ls = []
            sampled_names_ls = []

            for batch_id, (points, target, pnames, dly_idx) in tqdm(enumerate(test_loader_spl), total=len(test_loader_spl)):
                points, target = points.float().to(device), target.long().to(device)
                dly_idx = dly_idx.long().to(device)

                if args.use_pretrained_skeleton:
                    # prior sampling weight
                    ske_xyz, _, _ = model_ske(points, dly_idx)
                    min_dists, _ = mesh_utils.closest_distance_with_batch(points, ske_xyz, is_sum=False)

                    hist_edge = provider.linspace_nd(min_dists.min(dim=-1)[0], min_dists.max(dim=-1)[0], steps=args.bins+1)
                    hist_edge_mid = torch.mean(torch.stack([hist_edge[:, :-1], hist_edge[:, 1:]]), dim=0)
                    hist_edge_weight = torch.exp(-hist_edge_mid) / torch.sum(torch.exp(-hist_edge_mid), dim=-1, keepdim=True)

                    prior_weight = []
                    for i in range(min_dists.shape[0]):
                        hist_edge_i = hist_edge[i]
                        hist_edge_i[0] -= 0.1
                        hist_index_i = torch.bucketize(min_dists[i], hist_edge_i)
                        hist_index_i -= 1

                        hist_edge_weight_i = hist_edge_weight[i]
                        prior_weight.append(torch.index_select(hist_edge_weight_i, 0, hist_index_i))

                    prior_weight = torch.stack(prior_weight, dim=0).detach()
                else:
                    # for testing
                    prior_weight = torch.ones(points.size(0), points.size(1), dtype=torch.float).to(device).detach()

                points = points.transpose(1, 2)
                sampled_features, attn, stat_joint, stat_margin = model_spl(points, temp, hard, prior_weight)
                sampled_points = torch.matmul(attn, points.transpose(1, 2))

                # expec_joint = (-F.softplus(-stat_joint)).mean()
                # expec_margin = F.softplus(stat_margin).mean()
                # mutual_info = expec_joint - expec_margin
                # test_loss = -mutual_info

                # classification
                pred = model_cls(sampled_points.transpose(1, 2))

                pred_choice = pred.max(1)[1]
                correct = pred_choice.eq(target).sum()
                total_correct += correct.item()
                total_seen += target.size()[0]

                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat]).sum()
                    class_acc[cat, 0] += classacc.item()
                    class_acc[cat, 1] += target[target == cat].size()[0]

                # save sampled points
                sampled_points = sampled_points.cpu().detach().numpy()
                for idx in range(len(points)):
                    sampled_points_ls.append(sampled_points[idx])
                    sampled_names_ls.append(pnames[idx])

                if (epoch + 1) % args.save_spl_epoch == 0 or (epoch + 1) == spl_epoch:
                    sample_dir = spl_test_dir.joinpath('epoch %d' % (epoch + 1))
                    sample_dir.mkdir(exist_ok=True)

                    provider.save_ply_batch(sampled_points, sample_dir, names=pnames)

            instance_acc = total_correct / float(total_seen)
            class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
            class_acc = float(np.mean(class_acc[:, 2]))

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

                state = {'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
                         'model_state_dict': model_spl.state_dict(), 'optimizer_state_dict': optimizer_spl.state_dict()}
                savepath = os.path.join(str(spl_checkpoint_dir), 'sampling_best.pth')
                torch.save(state, savepath)
                logger.info('Saving at %s' % savepath)

                sample_dir = spl_test_dir.joinpath('best_points')
                sample_dir.mkdir(exist_ok=True)

                provider.save_ply_batch(sampled_points_ls, sample_dir, names=sampled_names_ls)

            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

        global_epoch += 1
        logger.info('Epoch time %f(m)' % ((datetime.datetime.now() - starttime).total_seconds() / 60.0))

    logger.info('End of sampling training...')


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''DATA LOADING'''
    print('Load dataset ...')

    # ------------------ ModelNet -----------------
    data_path = 'data/modelnet40_normal_resampled/'
    sample_path = 'log/sampling_2022-11-11-11-20-24/test_sampling/best_points/'
    test_dataset = ModelNetTestDataLoader(root=data_path, sampled_path=sample_path, args=args)
    testDataLoader_spl = torch.utils.data.DataLoader(test_dataset, batch_size=args.spl_batch_size, shuffle=False, num_workers=4)

    # -------------------- Task network --------------------
    # classification
    model_cls = PointNet(args).to(device)

    try:
        checkpoint = torch.load('./classification/log/pointnet_2022-10-24-20-25_seed_390/best_model.pth')
        model_cls.load_state_dict(checkpoint['model_state_dict'])
        print('Load pretrained classification model')
    except:
        print('Error: No existing classification model')
        exit()

    with torch.no_grad():
        model_cls.eval()

        total_correct = 0
        total_seen = 0
        class_acc = np.zeros((args.num_class, 3)).astype(float)

        for batch_id, (points, target) in tqdm(enumerate(testDataLoader_spl), total=len(testDataLoader_spl)):
            points, target = points.float().to(device), target.long().to(device)

            # classification
            pred = model_cls(points.transpose(1, 2))

            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(target).sum()
            total_correct += correct.item()
            total_seen += target.size()[0]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat]).sum()
                class_acc[cat, 0] += classacc.item()
                class_acc[cat, 1] += target[target == cat].size()[0]

        instance_acc = total_correct / float(total_seen)
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = float(np.mean(class_acc[:, 2]))

        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()

    if not args.test:
        train(args)
    else:
        test(args)
