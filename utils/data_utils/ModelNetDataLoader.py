import os
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial import Delaunay
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def delaunay_graph(point_set, ndelaunay):
    tri = Delaunay(point_set)
    indptr, indices = tri.vertex_neighbor_vertices

    G = nx.Graph()
    for i in range(len(point_set)):
        ind_i = list(indices[indptr[i]:indptr[i + 1]])
        edges = list(zip([i] * len(ind_i), ind_i))
        G.add_edges_from(edges)

    # neighboring
    k_path = 5
    min_neighbors = ndelaunay
    final_neighbors = ndelaunay
    neighbors_ls = []
    for i in range(len(point_set)):
        k_path_nodes = nx.single_source_shortest_path_length(G, i, cutoff=k_path)
        while len(k_path_nodes) < min_neighbors + 1:
            k_path = k_path + 1
            k_path_nodes = nx.single_source_shortest_path_length(G, i, cutoff=k_path)
        k_path = 5
        k_path_neighbors = list(k_path_nodes.keys())

        point_i = point_set[i]
        point_i = point_i[None, :]
        point_k = point_set[k_path_neighbors]

        dist = np.square(point_i - point_k)
        dist = np.sum(dist, axis=-1)
        dist_ind = np.argpartition(dist, final_neighbors + 1)

        dist_f = dist[dist_ind[:final_neighbors + 1]]
        neighbors_f = np.array(k_path_neighbors)[dist_ind[:final_neighbors + 1]]
        k_path_neighbors_final = neighbors_f[np.argsort(dist_f)]

        neighbors_ls.append(k_path_neighbors_final)

    return np.array(neighbors_ls)


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point
        self.ndelaunay = args.num_delaunay
        self.split = split
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.use_delaunay = args.use_delaunay
        self.num_category = args.num_class

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        self.list_of_points = []
        self.list_of_labels = []

        for index in tqdm(range(len(self.datapath))):
            fn = self.datapath[index]
            cls = self.classes[fn[0]]
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            self.list_of_points.append(point_set)
            self.list_of_labels.append(int(cls))

        if self.use_delaunay:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time) ...' % self.save_path)
                self.list_of_dlyidx = []

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    self.list_of_dlyidx.append(delaunay_graph(self.list_of_points[index][:, 0:3], self.ndelaunay))

                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.list_of_dlyidx, f)
            else:
                print('Load processed data from %s ...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_dlyidx = pickle.load(f)
        else:
            self.list_of_dlyidx = [-1] * len(self.datapath)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        # if self.process_data:
        #     point_set, label, dly = self.list_of_points[index], self.list_of_labels[index], self.list_of_dlyidx
        #     fn = self.datapath[index]
        # else:
        #     fn = self.datapath[index]
        #     cls = self.classes[fn[0]]
        #     label = np.array([cls]).astype(np.int32)
        #     point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        #
        #     if self.uniform:
        #         point_set = farthest_point_sample(point_set, self.npoints)
        #     else:
        #         point_set = point_set[0:self.npoints, :]

        point_set, label, dly = self.list_of_points[index], self.list_of_labels[index], self.list_of_dlyidx[index]
        fn = self.datapath[index]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label, os.path.basename(fn[1])[:-4], dly


class ModelNetTestDataLoader(Dataset):
    def __init__(self, root, sampled_path, args, split='test'):
        self.root = root
        self.sampled_path = sampled_path
        self.num_category = args.num_class

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.sampled_path, shape_ids[split][i]) + '.ply') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        label = np.array([cls]).astype(np.int32)

        ply_set = PlyData.read(fn[1])
        point_set = np.vstack([ply_set['vertex']['x'], ply_set['vertex']['y'], ply_set['vertex']['z']]).T
        point_set = np.array(point_set).astype(np.float32)

        return point_set, label[0]


if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser('training')

    parser.add_argument('--num_class', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_delaunay', type=int, default=32, help='Delaunay')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_delaunay', action='store_false', default=True, help='use delaunay neighbors')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    args = parser.parse_args()

    data = ModelNetDataLoader('../../data/modelnet40_normal_resampled/', args=args, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)
    for point, label, name, dly in DataLoader:
        print(point.shape)
        print(label.shape)
        print(name)
        print(dly.shape)
        print(dly)
        exit()

    data = ModelNetTestDataLoader('../../data/modelnet40_normal_resampled/', '../../log/sampling_2022-11-06-23-11-16/test_sampling/best_points/', args=args)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
        exit()
