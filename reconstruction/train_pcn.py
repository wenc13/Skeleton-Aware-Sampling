import os
import sys
import torch
import numpy as np
import argparse
import random
import datetime

from pathlib import Path
from tqdm import tqdm
from models.pcn import PCN, ChamferLoss

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import provider
from utils.logger import Logger
from utils.data_utils.ModelNetDataLoader import ModelNetDataLoader

# CUDA_VISIBLE_DEVICES=0 python train_pcn.py --epoch 300


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Training')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pcn', help='model name [default: pcn]')
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
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--detailed_output', action='store_true', default=False, help='return detailed output')
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
        exp_dir = exp_dir.joinpath('pcn' + '_' + timestr + '_seed_' + str(args.seed))
    else:
        exp_dir = exp_dir.joinpath(args.log_dir + '_' + timestr + '_seed_' + str(args.seed))
    exp_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = Logger('%s/%s.txt' % (exp_dir, 'pcn'))
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    data_path = '../data/modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PCN(emb_dims=args.emb_dims, detailed_output=args.detailed_output).to(device)
    criterion = ChamferLoss

    if args.use_pretrain_model:
        try:
            checkpoint = torch.load(os.path.join(args.resume, 'best_model.pth'))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Use pretrained model')
        except:
            logger.info('Error: No existing model')
            exit()
    else:
        start_epoch = 0
        logger.info('Starting training from scratch...')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    global_step = 0

    best_test_loss = np.inf

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        total_loss = 0.

        model.train()

        for batch_id, (points, _, _, _) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # points = torch.Tensor(points)
            # points = points.transpose(2, 1)

            points = points.float().to(device)
            output = model(points)
            loss = criterion(points, output['coarse_output'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            global_step += 1

        scheduler.step()

        train_loss = total_loss / float(batch_id + 1)
        logger.info('Train Loss: %f' % train_loss)

        with torch.no_grad():
            model.eval()

            total_loss = 0.
            recon_output = []
            recon_names = []

            for batch_id, (points, _, pnames, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points = points.float().to(device)
                output = model(points)
                loss = criterion(points, output['coarse_output'])

                total_loss += loss.item()
                recon_output.append(output['coarse_output'].cpu().numpy())
                recon_names += pnames

            test_loss = total_loss / float(batch_id + 1)

            if (epoch + 1) % 50 == 0:
                recon_dir = exp_dir.joinpath('epoch_' + str(epoch + 1))
                recon_dir.mkdir(exist_ok=True)
                provider.save_ply_batch(np.concatenate(recon_output), recon_dir, names=recon_names)

            if best_test_loss >= test_loss:
                best_test_loss = test_loss
                best_epoch = epoch + 1

                state = {'epoch': best_epoch, 'test_loss': test_loss,
                         'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                savepath = str(exp_dir) + '/best_model.pth'
                torch.save(state, savepath)
                logger.info('Saving at %s' % savepath)

                best_recon_dir = exp_dir.joinpath('best_recon')
                best_recon_dir.mkdir(exist_ok=True)
                provider.save_ply_batch(np.concatenate(recon_output), best_recon_dir, names=recon_names)

            logger.info('Test Loss: %f' % test_loss)
            logger.info('Best Test Loss: %f' % best_test_loss)

            # if best_test_loss >= test_loss:
            #     state = {'epoch': best_epoch, 'test_loss': test_loss,
            #              'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            #     savepath = str(exp_dir) + '/best_model.pth'
            #     torch.save(state, savepath)
            #     logger.info('Saving at %s' % savepath)

            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    main()
