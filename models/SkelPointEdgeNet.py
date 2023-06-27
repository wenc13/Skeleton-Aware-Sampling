import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mesh_utils
from utils import provider
from models.Sampling import get_graph_feature


class SkelPointEdgeNet(nn.Module):

    def __init__(self, args):

        super(SkelPointEdgeNet, self).__init__()
        self.k = args.k
        self.ndelaunay = args.num_delaunay
        self.use_normals = args.use_normals
        self.use_delaunay = args.use_delaunay
        self.num_skeleton_point = args.num_skeleton_point

        in_channel = 6 if args.use_normals else 3

        # Edge conv
        self.conv1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2))

        # Embedding
        self.conv5 = nn.Sequential(nn.Conv1d(64+64+128+256, args.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims), nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(args.emb_dims+64+64+128+256, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2))

        # delaunay features
        mlp = [384, 256, 256, 128, args.num_delaunay + 1]

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = 512 + in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def compute_loss(self, shape_xyz, skel_xyz, skel_radius, A, w1, w2, w3=0, lap_reg=False):

        bn = skel_xyz.size()[0]
        shape_pnum = float(shape_xyz.size()[1])
        skel_pnum = float(skel_xyz.size()[1])

        # sampling loss
        e = 0.57735027
        sample_directions = torch.tensor(
            [[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e], [-e, -e, -e]])
        sample_directions = torch.unsqueeze(sample_directions, 0)
        sample_directions = sample_directions.repeat(bn, int(skel_pnum), 1).cuda()
        sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
        sample_radius = torch.repeat_interleave(skel_radius, 8, dim=1)
        sample_xyz = sample_centers + sample_radius * sample_directions

        cd_sample1 = mesh_utils.closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
        cd_sample2 = mesh_utils.closest_distance_with_batch(shape_xyz, sample_xyz) / shape_pnum
        loss_sample = cd_sample1 + cd_sample2

        # point2sphere loss
        skel_xyzr = torch.cat((skel_xyz, skel_radius), 2)
        cd_point2pshere1 = mesh_utils.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
        cd_point2sphere2 = mesh_utils.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
        loss_point2sphere = cd_point2pshere1 + cd_point2sphere2

        # radius loss
        loss_radius = - torch.sum(skel_radius) / skel_pnum

        # Laplacian smoothness loss
        loss_smooth = 0
        if lap_reg:
            loss_smooth = self.get_smoothness_loss(skel_xyzr, A) / skel_pnum

        # loss combination
        final_loss = loss_sample + loss_point2sphere * w1 + loss_radius * w2 + loss_smooth * w3

        return final_loss

    def get_smoothness_loss(self, skel_xyz, A, k=6):

        bn, pn, p_dim = skel_xyz.size()[0], skel_xyz.size()[1], skel_xyz.size()[2]

        if A is None:
            knn_min = mesh_utils.knn_with_batch(skel_xyz, skel_xyz, k, is_max=False)
            A = torch.zeros((bn, pn, pn)).float().cuda()
            for i in range(bn):
                for j in range(pn):
                    A[i, j, knn_min[i, j, 1:k]] = 1
                    A[i, knn_min[i, j, 1:k], j] = 1

        neighbor_num = torch.sum(A, dim=2, keepdim=True)
        neighbor_num = neighbor_num.repeat(1, 1, p_dim)

        dist_sum = torch.bmm(A, skel_xyz)
        dist_avg = torch.div(dist_sum, neighbor_num)

        lap = skel_xyz - dist_avg
        lap = torch.norm(lap, 2, dim=2)
        loss_smooth = torch.sum(lap)

        return loss_smooth

    def compute_loss_pre(self, shape_xyz, skel_xyz):

        cd1 = mesh_utils.closest_distance_with_batch(shape_xyz, skel_xyz)
        cd2 = mesh_utils.closest_distance_with_batch(skel_xyz, shape_xyz)
        loss_cd = cd1 + cd2
        loss_cd = loss_cd * 0.0001

        return loss_cd

    def init_graph(self, shape_xyz, skel_xyz, valid_k=8):

        bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]

        knn_skel = mesh_utils.knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)
        knn_sp2sk = mesh_utils.knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)

        A = torch.zeros((bn, pn, pn)).float().cuda()

        # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
        A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
        A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1

        # initialize A with topology prior
        A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:2]] = 1
        A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:2], torch.arange(pn)[None, :, None]] = 1

        # valid mask: known existing links + knn links
        valid_mask = copy.deepcopy(A)
        valid_mask[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:valid_k]] = 1
        valid_mask[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:valid_k], torch.arange(pn)[None, :, None]] = 1

        # known mask: known existing links + known absent links, used as the mask to compute binary loss
        known_mask = copy.deepcopy(A)
        known_indice = list(range(valid_k, pn))
        known_mask[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, known_indice]] = 1
        known_mask[torch.arange(bn)[:, None, None], knn_skel[:, :, known_indice], torch.arange(pn)[None, :, None]] = 1

        return A, valid_mask, known_mask

    def split_point_feature(self, pc):

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, input_pc, delaunay_idx=None, compute_graph=False):
        # xyz, features = self.split_point_feature(input_pc)

        # obtain the sampled points and contextural features
        # for pointnet_module in self.SA_modules:
        #     xyz, features = pointnet_module(xyz, features)
        # sample_xyz, context_features = xyz, features

        x = input_pc.transpose(2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)

        # First edge conv
        x = get_graph_feature(x, k=self.k, use_normals=self.use_normals)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # Second edge conv
        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # Third edge conv
        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        # Fourth edge conv
        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, emb_dims, num_points)
        x = torch.cat((x, x1, x2, x3, x4), dim=1)  # (batch_size, emb_dims+64+64+128+256, num_points)
        x = self.conv6(x)  # (batch_size, emb_dims+64+64+128+256, num_points) -> (batch_size, 512, num_points)
        x = self.conv7(x)  # (batch_size, 512, num_points) -> (batch_size, 512, num_points)

        xyz_features = x

        # surface features
        # shape_cmb_features = torch.sum(weights[:, None, :, :] * context_features[:, :, None, :], dim=3)
        # shape_cmb_features = shape_cmb_features.transpose(1, 2)
        sample_xyz, context_features, weights, shape_cmb_features = None, None, None, None

        fps_idx = provider.farthest_point_sample(input_pc, self.num_skeleton_point)  # [B, npoint]
        new_xyz = provider.index_points(input_pc, fps_idx)  # [B, npoint, C]
        if self.use_delaunay:
            dly_idx = provider.index_points(delaunay_idx, fps_idx)
        else:
            dly_idx = mesh_utils.knn_with_batch(new_xyz, input_pc, self.ndelaunay + 1, is_max=False)

        grouped_xyz = provider.index_points(input_pc, dly_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(batch_size, self.num_skeleton_point, 1, input_pc.size(2))
        grouped_features = provider.index_points(xyz_features.transpose(1, 2), dly_idx)
        new_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)  # [B, npoint, nsample, C+D]
        new_features = new_features.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features = F.relu(bn(conv(new_features)))
        new_features = torch.max(new_features, 2, keepdim=True)[0]
        new_features = new_features.permute(0, 3, 1, 2)

        skel_xyz = torch.sum(new_features * grouped_xyz, dim=2)

        dists = torch.norm(grouped_xyz - skel_xyz[:, :, None, :], 2, dim=3)
        skel_r = torch.sum(new_features * dists[:, :, :, None], dim=2)

        # radii
        # min_dists, min_indices = mesh_utils.closest_distance_with_batch(sample_xyz, skel_xyz, is_sum=False)
        # skel_r = torch.sum(weights[:, :, :, None] * min_dists[:, None, :, None], dim=2)

        if compute_graph:
            A, valid_Mask, known_Mask = self.init_graph(input_pc[..., 0:3], skel_xyz)
            return skel_xyz, skel_r, sample_xyz, weights, shape_cmb_features, A, valid_Mask, known_Mask
        else:
            return skel_xyz, skel_r, shape_cmb_features
