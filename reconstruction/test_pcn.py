import os
import sys
import torch
import argparse

from tqdm import tqdm
from models.pcn import PCN, ChamferLoss

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_utils.ModelNetDataLoader import ModelNetDataLoader

# CUDA_VISIBLE_DEVICES=0 python test_pcn.py --log_dir


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Testing')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pcn', help='model name [default: pcn]')
    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_delaunay', type=int, help='Delaunay')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_true', default=False, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--detailed_output', action='store_true', default=False, help='return detailed output')

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

    model = PCN(emb_dims=args.emb_dims, detailed_output=args.detailed_output).to(device)
    criterion = ChamferLoss

    try:
        checkpoint = torch.load(str(exp_dir) + '/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        raise ValueError('No existing model')

    with torch.no_grad():
        model.eval()

        total_loss = 0.

        for batch_id, (points, _, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points = points.float().to(device)
            output = model(points)
            loss = criterion(points, output['coarse_output'])

            total_loss += loss.item()

        test_loss = total_loss / float(batch_id + 1)

        print('Test Loss: %f' % test_loss)


if __name__ == '__main__':
    main()
