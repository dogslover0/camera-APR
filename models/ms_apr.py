import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from efficientnet_pytorch import EfficientNet


# multi-scene absolute pose regression class
class MS_APR(nn.Module):
    def __init__(self, backbone_path, config):
        # load backbone from EfficientNet
        super(MS_APR, self).__init__()
        # self.backbone = EfficientNet.from_pretrained(backbone_path)
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        state_dict = torch.load(backbone_path)
        self.backbone.load_state_dict(state_dict)
        backbone_out = config.get("backbone_out_dim") # 1280
        backbone_next_out = config.get("backbone_next_out") # 1024
        n_scenes = config.get("num_scenes")
        # create layers using torch
        self.fc_after_fm = nn.Linear(backbone_out, backbone_next_out) # 1280,1024
        # build N fc layers for N different scenes
        # self.scene_fc_layers = []
        # for i in range(n_scenes):
        #     self.scene_fc_layers.append(nn.Linear(backbone_next_out, 7))
        self.scene_fc_layers = nn.Linear(backbone_next_out, 7)
        self.scene_cls = nn.Linear(backbone_next_out, n_scenes)    # scene classification
        
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1)

        # initialize FC Layer
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    # img-> CNN backbone -> feature -> scene_cls -> selected_fc_layer with feature -> pose regression
    def forward(self, data):
        # extract feature using CNN backbone -> avgpool
        feature = self.backbone.extract_features(data.get("img"))
        x = self.avg_pool_2d(feature)
        flattened = x.flatten(start_dim=1)
        # feature dim from backbone_out_dim to backbone_next_out dim,
        fm = self.dropout(F.relu(self.fc_after_fm(flattened)))
        # scene_fc_branch 场景相关fc层
        est_scene_distr = F.log_softmax(self.scene_cls(fm), dim=1)
        # est_scene_distr = est_scene_lsoft.argmax(1)

        # 根据场景分类索引，选择相应的fc层估计位姿
        # select the scene related layer
        sr_est_pose = self.scene_fc_layers(fm) # CHECK HERE
        return {"sr_est_pose": sr_est_pose, "est_scene_distr": est_scene_distr, "feature": fm}



