from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np


'''
This code is from 'Learning multi-scene camera pose with transformer' paper, 2021
CameraPose.py 从给定的数据path，和标注文件path，transform参数，和equalize_scenes创建实例，
imgs_paths, poses, scenes, scenes_ids属性，支持切片操作, camerapose[key] 通过__getitem__方法
'''

# a class inherited from torch.utils.data.Dataset
class CameraPoseDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None, equalize_scenes=False):
        """
        :param dataset_path: (str) the path to the dataset  数据集path
        :param labels_file: (str) a file with images and their path labels 标注文件path
        :param data_transform: (Transform object) a torchvision transform object torchvision 预处理方法
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        # 调用read_labels_file方法，返回图片路径，位姿，场景，场景id
        self.img_paths, self.poses, self.scenes, self.scenes_ids = read_labels_file(labels_file, dataset_path)
        self.dataset_size = self.poses.shape[0] # 数据集大小 pose数组(dataset_size,7)
        self.num_scenes = np.max(self.scenes_ids) + 1  # 场景数为scenes_ids中最大的id加1，因为从0开始
        # 获得多场景的图片idx，得到一个n维数组，n维场景数，每个数组元素都是该场景图片全部场景图片下的索引idx
        self.scenes_sample_indices = [np.where(np.array(self.scenes_ids) == i)[0] for i in range(self.num_scenes)]
        # 各场景图片占总图片的比例，作为抽样概率
        self.scene_prob_selection = [len(self.scenes_sample_indices[i])/len(self.scenes_ids)
                                     for i in range(self.num_scenes)]

        # 如果传入多场景数据集和标注，且equalize=True，
        if self.num_scenes > 1 and equalize_scenes:
            # 单个场景中包含图片数量最多的number
            max_samples_in_scene = np.max([len(indices) for indices in self.scenes_sample_indices])
            # 真实的多场景图片数量
            unbalanced_dataset_size = self.dataset_size
            # 按最大的单数据集包含数量*n(场景数)，计算抽样概率
            self.dataset_size = max_samples_in_scene*self.num_scenes
            # 多出来的图片
            num_added_positions = self.dataset_size - unbalanced_dataset_size
            # gap of each scene to maximum / # of added fake positions
            self.scene_prob_selection = [ (max_samples_in_scene-len(self.scenes_sample_indices[i]))/num_added_positions for i in range(self.num_scenes) ]
        self.transform = data_transform

    def __len__(self):
        return self.dataset_size


    # NOTE: __getitem__方法，可以将对象进行切片操作，在本方法中，创建CameraPoseDataset实例p，便可以通过p[key]的方式，取实例对象的子集。
    def __getitem__(self, idx):
        
        # 根据上面确定的采样概率，随机采样
        if idx >= len(self.poses): # sample from an under-repsented scene
            sampled_scene_idx = np.random.choice(range(self.num_scenes), p=self.scene_prob_selection)
            idx = np.random.choice(self.scenes_sample_indices[sampled_scene_idx])

        img = imread(self.img_paths[idx])
        pose = self.poses[idx]
        scene = self.scenes_ids[idx]
        if self.transform:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose, 'scene': scene}
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)   # 读取label文件，csv格式
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values] # 获取imgpath信息从csv文件中
    scenes = df['scene'].values # 场景信息
    scene_unique_names = np.unique(scenes) # 去除重复值，得到n场景名数组
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names))))) # 建立场景名和id字典
    scenes_ids = [scene_name_to_id[s] for s in scenes] # 场景id数组，通过scene_name_to_id字典键值对映射
    # 创建poses数组，维度(dataset_size,7)
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids


