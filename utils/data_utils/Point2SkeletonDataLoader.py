import os
from torch.utils.data import Dataset
from utils.provider import load_data_id, load_ply_points


class Point2SkeletonDataset(Dataset):
    def __init__(self, data_folder, point_num, split):
        self.data_folder = data_folder
        self.point_num = point_num

        assert (split == 'train' or split == 'test')
        if split == 'train':
            self.data_id = load_data_id(os.path.join(data_folder, 'all-train.txt'))
        else:
            self.data_id = load_data_id(os.path.join(data_folder, 'all-test.txt'))

    def __getitem__(self, index):
        data_pc = load_ply_points(os.path.join(self.data_folder, 'pointclouds', self.data_id[index] + '.ply'), expected_point=self.point_num)
        return data_pc, index

    def __len__(self):
        return len(self.data_id)


if __name__ == '__main__':
    data = Point2SkeletonDataset('../../data/point2skeleton/', point_num=1024, split='test')
    print(len(data))
