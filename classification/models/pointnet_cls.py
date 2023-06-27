import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        channel = 6 if args.use_normals else 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.trans_feat = None

    def forward(self, xyz, retrieval=False):
        x, trans, trans_feat = self.feat(xyz)
        self.trans_feat = trans_feat
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        ret_vec = x

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        if retrieval:
            return x, ret_vec
        else:
            return x

    def loss_function(self, pred, target):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(self.trans_feat)
        mat_diff_loss_scale = 0.001

        total_loss = loss + mat_diff_loss * mat_diff_loss_scale
        return total_loss
