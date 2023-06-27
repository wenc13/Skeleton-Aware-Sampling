import os
import sys
import torch
import numpy as np
import argparse

from tqdm import tqdm
from models.dgcnn import DGCNN
from models.pointnet_cls import PointNet
from models.pointnet2_cls_ssg import PointNet2

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils.ModelNetDataLoader import ModelNetDataLoader

# CUDA_VISIBLE_DEVICES=0 python test_cls.py --model pointnet --log_dir pointnet_2022-10-24-20-25_seed_390


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Testing')

    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name [default: dgcnn, pointnet, pointnet2]')
    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_delaunay', type=int, help='Delaunay')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_true', default=False, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')

    return parser.parse_args()


def main():
    args = parse_args()

    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    exp_dir = './log/' + args.log_dir
    if not os.path.exists(exp_dir):
        raise ValueError('No file folder exists')

    '''DATA LOADING'''
    print('Load dataset ...')

    data_path = '../data/modelnet40_normal_resampled/'
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
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

    try:
        checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
    except:
        raise ValueError('No existing model')

    with torch.no_grad():
        classifier.eval()

        total_correct = 0
        total_seen = 0
        class_acc = np.zeros((args.num_class, 3)).astype(float)

        for _, (points, target, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], args.num_class).to(device)

            for _ in range(args.num_votes):
                pred = classifier(points)
                vote_pool += pred
            pred = vote_pool / args.num_votes
            pred_choice = pred.data.max(1)[1]
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

        print('Test Instance Accuracy: %f, Test Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    main()
