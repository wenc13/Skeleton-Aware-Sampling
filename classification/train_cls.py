import os
import sys
import torch
import numpy as np
import argparse
import random
import datetime

from pathlib import Path
from tqdm import tqdm
from models.dgcnn import DGCNN
from models.pointnet_cls import PointNet
from models.pointnet2_cls_ssg import PointNet2

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import provider
from utils.logger import Logger
from utils.data_utils.ModelNetDataLoader import ModelNetDataLoader

# CUDA_VISIBLE_DEVICES=0 python train_cls.py --model pointnet --epoch 300


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Training')

    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', type=str, default='pointnet', help='model name [default: dgcnn, pointnet, pointnet2]')
    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--num_delaunay', type=int, help='Delaunay')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_true', default=False, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_pretrain_model', action='store_true', default=False, help='use pretrain model')
    parser.add_argument('--resume', type=str, default='', help='path to pretrained checkpoint')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--seed', type=int, help='Random seed.')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is None:
        args.seed = np.random.randint(1, 10000)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(args.model + '_' + timestr + '_seed_' + str(args.seed))
    else:
        exp_dir = exp_dir.joinpath(args.log_dir + '_' + timestr + '_seed_' + str(args.seed))
    exp_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = Logger('%s/%s.txt' % (exp_dir, args.model))
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    data_path = '../data/modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    assert isinstance(args.model, str)

    if args.model.lower() not in ['dgcnn', 'pointnet', 'pointnet2']:
        raise ValueError('Undefined model')
    else:
        if args.model.lower() == 'dgcnn':
            classifier = DGCNN(args).to(device)
        if args.model.lower() == 'pointnet':
            classifier = PointNet(args).to(device)
        if args.model.lower() == 'pointnet2':
            classifier = PointNet2(args).to(device)
    criterion = classifier.loss_function

    if args.use_pretrain_model:
        try:
            checkpoint = torch.load(os.path.join(args.resume, 'best_model.pth'))
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Use pretrained model')
        except:
            logger.info('Error: No existing model')
            exit()
    else:
        start_epoch = 0
        logger.info('Starting training from scratch...')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    global_step = 0

    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        total_correct = 0
        total_seen = 0

        classifier.train()

        for batch_id, (points, target, _, _) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.float().to(device), target.long().to(device)
            pred = classifier(points)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(target).sum()
            total_correct += correct.item()
            total_seen += target.size()[0]

            global_step += 1

        scheduler.step()

        train_instance_acc = total_correct / float(total_seen)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            classifier.eval()

            total_correct = 0
            total_seen = 0
            class_acc = np.zeros((args.num_class, 3)).astype(float)

            for _, (points, target, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)
                pred = classifier(points)

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

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

                state = {'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
                         'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                savepath = str(exp_dir) + '/best_model.pth'
                torch.save(state, savepath)
                logger.info('Saving at %s' % savepath)

            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    main()
