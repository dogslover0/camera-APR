import torch
import json
import numpy as np
import time
from models.ms_apr import MS_APR
from dataset.CameraPose import CameraPoseDataset
from util import utils


'''
在测试时，首先将测试图片feed给卷积部分提取特征，然后进行特征构图，放入GCN中学习特征间关系，最终得到优化后的位姿回归.
'''




class args:
    def __init__(self, dataset_path, label_path, model_name):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.model_name = model_name

args = args(dataset_path="../dataset_debug/7Scenes",
            label_path="../dataset_debug/chess_debug_train.csv",
            model_name='ms_apr')

with open('../config/config.json', "r") as f:
    config = json.load(f)
model_params = config[args.model_name]
general_params = config['general']
config = {**model_params, **general_params}



def feature_extract_net(ck_point, config):
    backbone_path = "../models/efficientnet-b0-355c32eb.pth"
    feature_extract_net = MS_APR(backbone_path, config)
    feature_extract_net.load_state_dict(torch.load(ck_point))
    return feature_extract_net


transform = utils.test_transforms.get('baseline')
equalize_scenes = config.get("equalize_scenes")
dataset = CameraPoseDataset(args.dataset_path, args.label_path, transform, equalize_scenes)
loader_params = {'batch_size': 1,
                 'shuffle': True,
                 'num_workers': 4}
dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

# 特征抽取建图
ck_point = "../out/7scenes_all_effnet_30epoches.pth"
print("Loading checkpoint from {}".format(ck_point))
feature_net = feature_extract_net(ck_point, config)     # 特征抽取网络
feature = []
img = []
tic = time.time()
fn = torch.nn.Linear(62720, 1024)
with torch.no_grad():
    for batch_idx, minibatch in enumerate(dataloader):
        img_input = minibatch.get('img')
        output_feature = feature_net.backbone.extract_features(img_input)
        output_feature = output_feature.flatten(start_dim=1)
        output_feature = fn(output_feature)
        feature.append(output_feature)
        img.append(img_input)

toc = time.time()

time_elpased = (toc - tic) / 1000
print("Feature extract using EfficientNet done! Time used:{:.4f}[s]".format(time_elpased))
# img: 存放和feature对应的img地址 feature: 图片经过CNN后的特征矩阵
# 计算特征间的L2距离，选择最近的2个结点建图，同时构建feature对应的label图谱



distr = []
head = 0
tail = 1

for i in range(head*100,tail*100):
    temp = []
    for j in range(head*100, tail*100):
        minus = feature[j] - feature[i]
        square = np.square(minus)
        square = square.numpy()
        final = np.sum(square, axis=1)
        # print(final.shape)
        temp[j] = np.sqrt(final)
        sorted_idx = np.argsort(temp)
        # 使用排序后的索引重构原数组
        for i in range(temp):
            distr[i] = temp(sorted_idx[i])
    if tail*100 < feature.shape[0]:
        tail += 1
        head += 1

print(distr.shape)

# 选择最接近的2个特征，做相邻结点,创建feature.shape[0]/100个小图
for idx in range(feature.shape[0]/100):
    temp = ['f'+str(i) for i in range(idx*100,(idx+1)*100)]
    nodes[idx] = temp

print(nodes)


