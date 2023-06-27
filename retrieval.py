import os
import sys
import torch
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classification'))

from tqdm import tqdm
from classification.models.pointnet_cls import PointNet
from utils.data_utils.ModelNetDataLoader import ModelNetTestDataLoader
from utils.analyze_precision_recall import calc_macro_mean_average_precision

# CUDA_VISIBLE_DEVICES=0 python retrieval.py


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_delaunay', type=int, default=32, help='Delaunay')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_false', default=True, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--spl_batch_size', type=int, default=4, help='batch size in training')

    return parser.parse_args()


def main():
    args = parse_args()

    '''HYPER PARAMETER'''
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

    ret_vec_ls, label_ls = [], []
    with torch.no_grad():
        model_cls.eval()

        total_correct = 0
        total_seen = 0
        class_acc = np.zeros((args.num_class, 3)).astype(float)

        for batch_id, (points, target) in tqdm(enumerate(testDataLoader_spl), total=len(testDataLoader_spl)):
            points, target = points.float().to(device), target.long().to(device)

            # classification
            pred, ret_vec = model_cls(points.transpose(1, 2), retrieval=True)

            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(target).sum()
            total_correct += correct.item()
            total_seen += target.size()[0]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat]).sum()
                class_acc[cat, 0] += classacc.item()
                class_acc[cat, 1] += target[target == cat].size()[0]

            ret_vec_ls.append(ret_vec.cpu().detach().numpy())
            label_ls.append(np.expand_dims(target.cpu().detach().numpy(), axis=-1))

        instance_acc = total_correct / float(total_seen)
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = float(np.mean(class_acc[:, 2]))

        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

        ret_vecs = np.vstack(ret_vec_ls)
        labels = np.vstack(label_ls)
        res_macro = calc_macro_mean_average_precision(ret_vecs, labels)
        print('Retrieval mAP: %f' % res_macro)


if __name__ == '__main__':
    main()
