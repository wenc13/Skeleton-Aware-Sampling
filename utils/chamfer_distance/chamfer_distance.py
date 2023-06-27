from torch import nn
from torch.autograd import Function
import torch
import importlib
import os

chamfer_found = importlib.util.find_spec("chamfer_dist") is not None

if not chamfer_found:
    # Cool trick from https://github.com/chrdiller
    print("Jitting chamfer distance")

    from torch.utils.cpp_extension import load

    chamfer_dist = load(name="chamfer_dist",
                        sources=["/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_distance.cpp"]),
                                 "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_distance.cu"])])
    print("Loaded JIT chamfer distance")

else:
    import chamfer_dist
    print("Loaded compiled chamfer distance")


class ChamferDistanceFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()

        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            chamfer_dist.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
            torch.cuda.set_device(device)

            chamfer_dist.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            chamfer_dist.backward(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )
        else:
            gradxyz1 = gradxyz1.to(device)
            gradxyz2 = gradxyz2.to(device)
            chamfer_dist.backward_cuda(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )

        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)
