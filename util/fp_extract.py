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

args = args(dataset_path="../dataset/7Scenes",
            label_path="../GTPose/7Scenes_train_val/train/fire_train.csv",
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
numpy_seed = 2
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_id = "cuda:0"
np.random.seed(numpy_seed)
device = torch.device(device_id)


ck_point = "../out/7scenes_all_effnet_30epoches.pth"
print("Loading checkpoint from {}".format(ck_point))
tic=time.time()
feature_net = feature_extract_net(ck_point, config).to(device) # 特征抽取网络
toc=time.time()
print("load network params to device , time elapsed:{:.4f}[s]".format(toc-tic))
feature = []    # 260,100,1024
pose = []       # 260,100,7
tic = time.time()



with torch.no_grad():
    for batch_idx, minibatch in enumerate(dataloader):
        for k, v in minibatch.items():
            minibatch[k] = v.to(device)  # 将数据传输给设备
        # img_input = minibatch.get('img').to(device)
        poses = minibatch.get('pose')
        pose.append(poses)
        feed_prop = feature_net(minibatch)
        output_feature = feed_prop.get("feature")
        feature.append(output_feature)


toc = time.time()

time_elpased = (toc - tic)
print("Feature extract using EfficientNet done! Time used:{:.4f}[s]".format(time_elpased))
# img: 存放和feature对应的img地址 feature: 图片经过CNN后的特征矩阵
# 计算特征间的L2距离，选择最近的2个结点建图，同时构建feature对应的label图谱


pass
distr = [] # for室内数据集, (260,100,100)
distr_sorted = []
head = 0
tail = 1

while (tail*100 < len(feature)):
    for i in range(head*100,tail*100):
        temp = np.zeros([100,100],dtype=float)
        edges_base = np.zeros([100,100],dtype=int)
        for j in range(head*100, tail*100):
            minus = (feature[j] - feature[i]).cpu().numpy()
            square = np.square(minus)
            # square = square.numpy() # convert feature:Tensor to ndarray
            final = np.sum(square, axis=1)
            # print(final.shape)
            temp[i][j] = np.sqrt(final)
            temp[j][i] = temp[i][j]
        # edges_base is the row-wised sorted index 存放每一行的distr的按序排列索引
        edges_base[i][:] = np.argsort(temp[i][:], axis=0) # NOTE: CHECK the axis
    head += 1
    tail += 1
    # 使用排序后的索引重构原数组
    distr_sorted[head] = edges_base
    distr[head] = temp

#NOTE check 'distr []'

pass

nodes = [] #260,100,1
# 选择最接近的2个特征，做相邻结点,创建feature.shape[0]/100个小图
for idx in range(feature.shape[0]/100):
    temp = ['f'+str(i) for i in range(idx*100,(idx+1)*100)]
    nodes[idx] = temp

print(nodes)
#NOTE: check 'nodes []'

edges = [] #260,100,200 each node has two neibor nodes
for idx in range(feature.shape[0]/100):
    select_1, select_2 = distr[idx][0], distr[idx][1]
