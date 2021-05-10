import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    
    def __init__(self, config):
        super(PoseLoss, self).__init__()
        self.learnable = config.get("learnable")
        self.s_x = torch.nn.Parameter(torch.Tensor([config.get("s_x")]), requires_grad=self.learnable)
        self.s_q = torch.nn.Parameter(torch.Tensor([config.get("s_q")]), requires_grad=self.learnable)
        self.norm = config.get("norm")

    def forward(self, est_pose, gt_pose):

        # Position loss
        l_x = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=self.norm).mean()
        # Orientation loss (normalized to unit norm)
        l_q = torch.norm(F.normalize(gt_pose[:, 3:], p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                            dim=1, p=self.norm).mean()

        if self.learnable:
            return l_x * torch.exp(-self.s_x) + self.s_x + l_q * torch.exp(-self.s_q) + self.s_q  # Learnable loss
        else:
            return self.s_x*l_x + self.s_q*l_q      #