import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from faster_rcnn.nets.frcnn import FasterRCNN
from faster_rcnn.nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from faster_rcnn.utils.callbacks import eval, loss_record
from faster_rcnn.utils.dataloader import frcnn_data, frcnn_dataset_collate
from utils.utils import get_classes
from faster_rcnn.utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    classes_path    = '/home/hjh/PJ2/voc_classes.txt'
    model_path      = '/home/hjh/PJ2/para/voc_weights_vgg.pth'
    train_annotation_path   = '/home/hjh/PJ2/2007_train.txt'
    val_annotation_path     = '/home/hjh/PJ2/2007_val.txt'

    Cuda            = True
    train_gpu       = [0,1]
    input_shape     = [600, 600]

    base_sizes    = [8, 16, 32]
    epoch_num      = 80
    batch_size = 2
    
    # adam, 学习率下降方式:'cos'
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01

    #   momentum        优化器内部使用到的momentum参数
    momentum            = 0.9

    #   保存
    save_period         = 5
    save_dir            = 'logs_frcnn'
    eval_period         = 5

    #   获取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    
    model = FasterRCNN(num_classes, base_sizes = base_sizes)
    weights_init(model)

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)


    #   记录Loss
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = loss_record(log_dir, model, input_shape=input_shape)

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = False
        model_train = model_train.cuda()

    #   读取数据集对应的txt
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    wanted_step = 1.5e4
    total_step  = num_train // batch_size * epoch_num
    
    #   自适应调整学习率
    nbs             = 16
    lr_limit_max    = 1e-4 
    lr_limit_min    = 1e-4 
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    #   优化器
    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999))

    #   获得学习率下降
    lr_scheduler_func = get_lr_scheduler('cos', Init_lr_fit, Min_lr_fit, epoch_num)

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    train_dataset   = frcnn_data(train_lines, input_shape, train = True)
    val_dataset     = frcnn_data(val_lines, input_shape, train = False)

    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, pin_memory=True, 
                                drop_last=True, collate_fn=frcnn_dataset_collate)

    trainer      = FasterRCNNTrainer(model_train, optimizer)

    #   记录eval的map曲线
    eval   = eval(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, 
                                    eval_flag=True, period=eval_period)

    #   开始训练
    for epoch in range(epoch_num):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model, trainer, loss_history, eval, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, epoch_num, Cuda, save_period, save_dir)
