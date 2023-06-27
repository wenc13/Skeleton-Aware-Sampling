# https://github.com/vinits5/learning3d.git
import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.chamfer_distance.chamfer_distance import ChamferDistance
# from utils.chamfer_distance_ import dist_chamfer_3D
# distChamferL2 = dist_chamfer_3D.chamfer_3DDist()

# def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
#     from .chamfer_distance import ChamferDistance
#     cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
#     cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
#     cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
#     chamfer_loss = (cost_p0_p1 + cost_p1_p0) / 2.0
#     return chamfer_loss


# class ChamferDistanceLoss(nn.Module):
#     def __init__(self):
#         super(ChamferDistanceLoss, self).__init__()
#
#     def forward(self, template, source):
#         return chamfer_distance(template, source)


def ChamferLoss(target, prediction, reduction='mean'):
    # dist1, dist2, _, _ = distChamferL2(target, prediction)

    # if reduction == 'mean':
    #     loss = torch.mean(dist1) + torch.mean(dist2)
    # elif reduction == 'sum':
    #     loss = torch.sum(dist1) + torch.sum(dist2)
    # else:
    #     raise ValueError()

    # For fair comparison
    cost_p0_p1, cost_p1_p0, _, _ = ChamferDistance()(target, prediction)
    cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
    cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
    loss = (cost_p0_p1 + cost_p1_p0) / 2.0

    return loss


class PCN(nn.Module):
    def __init__(self, emb_dims=1024, num_coarse=1024, grid_size=4, detailed_output=False):
        super(PCN, self).__init__()
        self.emb_dims = emb_dims
        self.num_coarse = num_coarse
        self.detailed_output = detailed_output
        self.grid_size = grid_size
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        self.encoder_layers1 = nn.Sequential(nn.Conv1d(3, 128, 1),
                                             nn.ReLU(),
                                             nn.Conv1d(128, 256, 1))

        self.encoder_layers2 = nn.Sequential(nn.Conv1d(2 * 256, 512, 1),
                                             nn.ReLU(),
                                             nn.Conv1d(512, self.emb_dims, 1))

        self.decoder_layers = nn.Sequential(nn.Linear(self.emb_dims, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, self.num_coarse * 3))

        if detailed_output:
            self.folding_layers = nn.Sequential(nn.Conv1d(1029, 512, 1),
                                                nn.ReLU(),
                                                nn.Conv1d(512, 512, 1),
                                                nn.ReLU(),
                                                nn.Conv1d(512, 3, 1))

    def fine_decoder(self, global_feature_v, coarse_output):
        # Fine Output
        linspace = torch.linspace(-0.05, 0.05, steps=self.grid_size)
        grid = torch.meshgrid(linspace, linspace)
        grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2))  # 16x2
        grid = torch.unsqueeze(grid, dim=0)  # 1x16x2
        grid_feature = grid.repeat([coarse_output.shape[0], self.num_coarse, 1]).cuda()  # Bx16384x2

        point_feature = torch.unsqueeze(coarse_output, dim=2)                 # Bx1024x1x3
        point_feature = point_feature.repeat([1, 1, self.grid_size ** 2, 1])  # Bx1024x16x3
        point_feature = torch.reshape(point_feature, (-1, self.num_fine, 3))  # Bx16384x3

        global_feature = torch.unsqueeze(global_feature_v, dim=1)       # Bx1x1024
        global_feature = global_feature.repeat([1, self.num_fine, 1])   # Bx16384x1024

        feature = torch.cat([grid_feature, point_feature, global_feature], dim=2)  # Bx16384x1029

        center = torch.unsqueeze(coarse_output, dim=2)          # Bx1024x1x3
        center = center.repeat([1, 1, self.grid_size ** 2, 1])  # Bx1024x16x3
        center = torch.reshape(center, [-1, self.num_fine, 3])  # Bx16384x3

        output = feature.permute(0, 2, 1)
        output = self.folding_layers(output)
        fine_output = output.permute(0, 2, 1) + center

        return fine_output

    def encode(self, input_data):
        num_points = input_data.shape[-1]

        output = self.encoder_layers1(input_data)

        global_feature_g = torch.max(output, 2)[0].contiguous()

        global_feature_g = global_feature_g.unsqueeze(2)
        global_feature_g = global_feature_g.repeat(1, 1, num_points)
        output = torch.cat([output, global_feature_g], dim=1)

        output = self.encoder_layers2(output)

        global_feature_v = torch.max(output, 2)[0].contiguous()

        return global_feature_v

    def decode(self, global_feature_v):
        output = self.decoder_layers(global_feature_v)
        coarse_output = output.view(global_feature_v.shape[0], self.num_coarse, 3)

        return coarse_output

    def forward(self, input_data):
        input_data = input_data.transpose(2, 1)
        if input_data.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        global_feature_v = self.encode(input_data)
        coarse_output = self.decode(global_feature_v)

        result = {'coarse_output': coarse_output}

        if self.detailed_output:
            fine_output = self.fine_decoder(global_feature_v, coarse_output)
            result['fine_output'] = fine_output

        return result


if __name__ == '__main__':
    # Test the code.
    x = torch.rand((10, 3, 1024)).cuda()

    pcn = PCN(emb_dims=1024, detailed_output=True).cuda()
    y = pcn(x)
    print("Network Architecture: ")
    print(pcn)
    print("Input Shape of PCN: ", x.shape)
    print("Coarse Output Shape of PCN: ", y['coarse_output'].shape)
    print("Fine Shape of PCN: ", y['fine_output'].shape)
