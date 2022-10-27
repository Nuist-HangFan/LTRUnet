import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset
from torch.nn import init
from tensorboardX import SummaryWriter

from utils.makedataset import *
from utils.train_test import *
from utils.normalization import *
from model.LTRUnet import LTRUnet

def load_data(data_root, sample_list_root, select, out_layer, Tb_flag,hgt_flag,  time_flag, BATCH_SIZE, num_workers, shuffle=True):
    Train_list = read_txt(os.path.join(sample_list_root, "Train.txt"))
    Validation_list = read_txt(os.path.join(sample_list_root, "Validation.txt"))
    Test_list = read_txt(os.path.join(sample_list_root, "Test.txt"))
    if select:
        Train_list = select_data(Train_list, select)
    Train_set = Make_Dataset(Train_list, data_root, out_layer,Tb_flag,  hgt_flag, time_flag)
    Validation_set = Make_Dataset(Validation_list, data_root, out_layer, Tb_flag, hgt_flag, time_flag)
    Test_set = Make_Dataset(Test_list, data_root, out_layer,Tb_flag,  hgt_flag, time_flag)

    print('Size of Train Data:', len(Train_set.data_list))
    print('Size of Validationt Data:', len(Validation_set.data_list))
    print('Size of Test Data:', len(Test_set.data_list))

    Train_loader = Loader(Train_set, BATCH_SIZE, num_workers)
    Validation_loader = Loader(Validation_set, BATCH_SIZE, num_workers)
    Test_loader = Loader(Test_set, BATCH_SIZE, num_workers)

    return Train_loader, Test_loader, Validation_loader

# calculate the number of the input and output layers
def get_layer_num(Tb_flag, hgt_flag, time_flag, ):
    in_layer = 1
    out_layer = 13
    if Tb_flag:
        in_layer += 13
    if hgt_flag:
        in_layer += 1
    if time_flag:
        in_layer += 1
    print(in_layer, out_layer)
    return in_layer, out_layer

# set the  writter to record the parameters during training
def set_writer(work_root, exe_name):
    writer_path = os.path.join(work_root,'Tensorboard',exe_name)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(log_dir=writer_path)
    return writer

def main():
    # Set the hyperparamters (Need to modify)
    BATCH_SIZE = 128 
    EPOCHS = 300
    num_workers = 8
    exe_name = 'test'                  # Experiment name
    work_root = '~/LTRUnet'            # The path of the experiment
    data_root = '~/data/T_HDF_data'    # The path of data
    select = []                        # The list used to select specific samples.
    sample_list_root = '~/sample_list' # The path of the files stored sample lists of training, validatuion, and test dataset.
    best_save_location = None           # The path of the best parameters of model, if None, the model will be trained from scratch
    lr = 3e-4                            # Leaning rate
    weight_decay =0.005                  # Weight decay in optimizer
    
    # Switches for using auxiliary information (Need to modify)
    Tb_flag = False           
    hgt_flag = False
    time_flag = False

    # set the device
    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the writer.
    writer_path = os.path.join(work_root,'Tensorboard',exe_name)

    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(log_dir=writer_path)

    # Load dataset
    in_layer, out_layer = get_layer_num(Tb_flag, hgt_flag, time_flag,)
    Train_loader, Test_loader, Validation_loader = load_data(data_root, sample_list_root, select,out_layer,Tb_flag, hgt_flag, time_flag, BATCH_SIZE, num_workers)

    print(in_layer, out_layer)
    model = LTRUnet(in_layer, out_layer).to(DEVICE)

    # Train if is None
    if not best_save_location:
        model_save_root = os.path.join(work_root, 'model_save')
        if not os.path.exists(model_save_root):
            os.makedirs(model_save_root)
        best_save_location = os.path.join(model_save_root, exe_name + '_best.pt')
        Train_step(model, DEVICE, EPOCHS, BATCH_SIZE,lr, weight_decay,best_save_location, writer, Train_loader, Validation_loader)
        print("Finish")

    # Test
    Test_step(model, DEVICE, best_save_location, Test_loader)

if __name__ == "__main__":
    main()