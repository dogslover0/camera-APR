import torch
import numpy as np
import json
import time
import argparse
import logging
from os.path import join
from util import utils
from models.base_loss import PoseLoss
from models.gen_model import get_model
from dataset.CameraPose import CameraPoseDataset



if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser()
    # compulsory params
    # arg_parser.add_argument("model_name",
    #                         help="name of the model to call getmodel func")
    # arg_parser.add_argument("mode",help="train or eval")
    # arg_parser.add_argument("dataset_pth",
    #                         help="path of your dataset")
    # arg_parser.add_argument("label_pth",
    #                         help="label file for your dataset")
    # # arg_parser.add_argument("backbone_name",
    # #                         help="backbone name to get")
    # arg_parser.add_argument("backbone_path", help="backbone pth file path")
    # # optional param
    # arg_parser.add_argument("--checkpoint_pth",
    #                         help="checkpoint path to reload your previous training state")
    # args = arg_parser.parse_args()
    #NOTE:测试部分
    class args:
        def __init__(self, model_name, backbone_path, mode, label_path, dataset_path):
            self.model_name = model_name
            self.backbone_path = backbone_path
            self.mode = mode
            self.label_path = label_path
            self.dataset_path = dataset_path

    args = args(model_name="ms_apr",
                backbone_path="models/efficientnet-b0-355c32eb.pth",
                mode="train",
                label_path="dataset_debug/chess_debug_train.csv",
                dataset_path="dataset_debug/7Scenes/")
    # NOTE:测试部分
    utils.init_logger()

    # 记录实验details
    logging.info("Start with {} with {}".format(args.model_name, args.mode))
    logging.info("Using dataset from {}".format(args.dataset_path))
    logging.info("Using label file from {}".format(args.label_path))

    # 读取json配置
    with open('config/config.json', "r") as f:
        config = json.load(f)
    model_params = config[args.model_name]
    general_params = config['general']
    # NOTE:**kwargs 传入参数为字典类型
    config = {**model_params, **general_params}
    # NOTE: dict.items 返回可遍历的字典key和value
    logging.info("Running with configuration : \n{}".format(
        '\n'.join(["\t {}: {}".format(k, v) for k, v in config.items()])))
    
    # 设置随机数seed和运行的设备device
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device_id = "cpu" # 默认为CPU
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # 创建模型
    # print(args.backbone_name)
    model = get_model(config, args.model_name, args.backbone_path).to(device)
    # 如果设置了checkpoint，加载它
    # if args.checkpoint_pth:
    #     model.load_state_dict(torch.load(args.checkpoint_pth), map_location=device_id)
    #     logging.info("Intiializing from checkpoint {}".format(args.checkpoint_path))
    

    # 开始模型训练/测试
    if args.mode == "train":
        model.train()

        # LOSS
        poss_loss = PoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()

        # Optimizer and Scheduler
        params = list(model.parameters()) + list(poss_loss.parameters())
        # print(params)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                lr=config.get('lr'),
                                eps=config.get('eps'),
                                weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # 数据集 dataset & data loader
        
        # 不做数据增强(预处理)
        no_augment = config.get('no_augment')
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        equalize_scenes = config.get("equalize_scenes")
        dataset = CameraPoseDataset(args.dataset_path, args.label_path, transform, equalize_scenes)
        loader_params = {'batch_size': config.get('batch_size'),
                            'shuffle':True,
                            'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # 训练参数配置
        n_freq_print = config.get('n_freq_print')       # 每几个epoches打印一次结果
        n_freq_checkpoint = config.get('n_freq_checkpoint') # 每几个epoches保存一个checkpoint
        n_epoches = config.get('n_epochs')
        # 训练
        checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epoches):

            # 重置loss用于日志显示在屏幕上
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                # minibatch 包括 图片,gt位姿，gt场景
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device) #将数据传输给设备
                # 这里调用了CameraPose 的__getitem__方法，对象切片,得到img,pose,scene的字典
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_idx

                # 梯度清0
                optimizer.zero_grad()
                # 前向传播，估计位姿
                feed_forward = model(minibatch)
                
                est_pose = feed_forward.get('sr_est_pose')
                # CHECK HERE
                est_scene_log_distr = feed_forward.get('est_scene_distr')
                # feature_vec = feed_forward.get('feature')
                # print(feature_vec)
                if est_scene_log_distr is not None:
                    criterion = poss_loss(est_pose, gt_pose) + nll_loss(est_scene_log_distr, gt_scene)
                else:
                    criterion = poss_loss(est_pose, gt_pose)

                # 记录loss，画图
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                #反向传播
                criterion.backward()
                optimizer.step()

                if batch_idx % n_freq_print == 0:
                    # NOTE:detach() call this func to do backward propergation
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, " 
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                    batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                    posit_err.mean().item(),
                                                                    orient_err.mean().item()))
            # 保存断点 checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))
            
            # Scheduler更新学习率
            scheduler.step()
        logging.info('Training completed')
        final_ck_point = torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # 画图
        loss_fig_path = checkpoint_prefix + '_loss_fig.png'
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)





    else: # TEST 测试模式
        model.eval()

        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_pth, args.label_pth, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None # 测试时不使用scene标签

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # 前向传播
                tic = time.time()
                est_pose = model(minibatch)
                toc = time.time()

                # 测试error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inffered in {:.2f}[ms]".format(
                    stats[i, 0], stats[i, 1], stats[i, 2]))
                
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:,1])))
        logging.info("Mean inference time: {:.2f}[ms]".format(np.mean(stats[:, 2])))





                            


