import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k, with_normals=False):
    p = x[:, :3, :]
    n = x[:, 3:, :]

    inner = -2 * torch.matmul(p.transpose(2, 1), p)
    pp = torch.sum(p ** 2, dim=1, keepdim=True)
    pairwise_distance = -pp - inner - pp.transpose(2, 1)

    if with_normals:
        inner = -2 * torch.matmul(n.transpose(2, 1), n)
        n_pairwise_distance = -2 - inner
        pairwise_distance *= (1 + n_pairwise_distance)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, use_normals=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not use_normals:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x, k=k, with_normals=True)  # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class GumbelSampling(nn.Module):
    def __init__(self, in_features, sampling_n, activation_fn=None):
        super(GumbelSampling, self).__init__()
        self.sampler = nn.Linear(in_features, sampling_n)

    def forward(self, x, tau=1., hard=False, mask=None):
        sample_weights = self.sampler(x)

        # if mask is not None:
        #     NINF = -9e15
        #     sample_weights = sample_weights.masked_fill((1 - mask).unsqueeze(-1), NINF)

        batch_size, num_points, sampling_n = sample_weights.shape
        attn = F.gumbel_softmax(sample_weights.transpose(1, 2).contiguous().view(-1, num_points), tau=tau, hard=hard)
        attn = attn.view(batch_size, sampling_n, num_points)

        sampled = attn.matmul(x)
        return sampled.transpose(1, 2), attn


class StatisticsNet(nn.Module):
    """Statistics network"""
    def __init__(self, channel1, channel2):
        super(StatisticsNet, self).__init__()
        # self.embedding1 = nn.Linear(in_features=channel1, out_features=256)
        # self.embedding2 = nn.Linear(in_features=256, out_features=256)

        # self.embedding3 = nn.Linear(in_features=channel2, out_features=256)
        # self.embedding4 = nn.Linear(in_features=256, out_features=256)

        self.linear1 = nn.Linear(in_features=channel1 + channel2, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, repre1, repre2):
        x = torch.cat([repre1, repre2], dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        stat = self.linear3(x)

        return stat


class InfoSampling(nn.Module):
    def __init__(self, args):
        super(InfoSampling, self).__init__()
        self.k = args.k
        self.use_normals = args.use_normals

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
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))

        # Sampling
        self.dp = nn.Dropout(p=args.dropout)
        self.sampler = GumbelSampling(256, args.num_sampling)

        self.stat = StatisticsNet(256, 256)

    def forward(self, x, tau, hard, prior_weight):
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
        x = self.conv7(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)

        # orignal_repr = x.max(dim=-1)[0]
        # orignal_repr_shuffle = torch.cat([orignal_repr[1:], orignal_repr[:1]], dim=0)

        x = x * prior_weight.unsqueeze(1)
        x = self.dp(x)
        sampled, attn = self.sampler(x.transpose(1, 2), tau=tau, hard=hard)

        # mutual information
        # stat_joint = self.stat(orignal_repr, sampled_repr)
        # stat_margin = self.stat(orignal_repr_shuffle, sampled_repr)
        stat_joint, stat_margin = None, None

        return sampled, attn, stat_joint, stat_margin


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--num_sampling', type=int, default=512, metavar='N', help='Num of sampling points')

    args = parser.parse_args()

    model = InfoSampling(args)
    model(torch.rand(4, 3, 1024))
