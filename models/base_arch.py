import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Base_arch(nn.Module):
    """
    a class to implement a basic arch of APR task, with a changeable CNN backbone part
    """
    def __init__(self, backbone_name, config):
        super(Base_arch, self).__init__()

        # load the CNN backbone from torch
        # self.backbone = torch.load(backbone_name)
        self.backbone = EfficientNet.from_pretrained(backbone_name)
        backbone_out_dim = config.get("backbone_out_dim")
        regressor_in_dim = config.get("regressor_in_dim")
        
        # fc layer before two regressors
        self.fc1 = nn.Linear(backbone_out_dim, regressor_in_dim)
        self.rg1 = nn.Linear(regressor_in_dim, 3)
        self.rg2 = nn.Linear(regressor_in_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
        
    # images -> CNN backbone -> feature -> FC layers -> two regressors   
    def forward(self, data):
        x = self.backbone.extract_features(data.get("img"))
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        pose_x = self.rg1(x)
        pose_q = self.rg2(x)
        return {'pose': torch.cat((pose_x, pose_q), dim=1)}

