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


# Get dataloaders for training,testing and validation 
def load_data(data_root, sample_list_root, select, test_select, out_layer, Tb_flag,hgt_flag,  time_flag, BATCH_SIZE, num_workers, if_PGT = True,shuffle=True):
    Train_list = read_txt(os.path.join(sample_list_root, "Train.txt"))
    Validation_list = read_txt(os.path.join(sample_list_root, "Validation.txt"))
    Test_list = read_txt(os.path.join(sample_list_root, "Test.txt"))
    if if_PGT: # if using partial group training, the samples for training each group should be different from that for training temporal LTRUnet
        Train_list = select_data(Train_list, ['-10-00', '-30-00','-50-00'])  # Need to modify
    if select:
        Train_list = select_data(Train_list, select)
    if test_select:
        Test_list = select_data(Train_list, test_select)
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
    return in_layer, out_layer

# set the  writter to record the parameters during training
def set_writer(work_root, exe_name):
    writer_path = os.path.join(work_root,'Tensorboard',exe_name)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(log_dir=writer_path)
    return writer

# select samples for each group (Need to modify when use different groups)
def get_select_from_group(group_name):
    if group_name == 'night':
        select = ['_' + str(i+11).rjust(2,'0') + '-' for i in range(12)]
    elif group_name == 'day':
        select = ['_' + str(i+1).rjust(2,'0') + '-' for i in range(7)]
    elif int(group_name[:2]) == 0:
        select = ['_23-', '_00-', '_01-']
    else:
        select = ['_' + str(int(group_name[:2])+i-1).rjust(2,'0') + '-' for i in range(3)]
    return select


def main():
    # Set the hyperparamters (Need to modify)
    BATCH_SIZE = 64                      # Batch size for each group 
    EPOCHS = 100                         # Epoch for each group
    num_workers = 8
    exe_name = 'test_GT'                  # Experiment name
    work_root = '~/LTRUnet'            # The path of the experiment
    data_root = '~/data/T_HDF_data'    # The path of data
    select = []                        # The list used to select specific samples.
    sample_list_root = '~/sample_list' # The path of the files stored sample lists of training, validatuion, and test dataset.
    basic_save_location = '/public/home/FanHang/final_job/basic_model/model_save/Tb_half_best.pt'  # PGT: The path of the best parameters of the temporal LTRUnet; IGT: The path of the initial parameters of LRTUnet.
    lr = 6e-5                            # Leaning rate
    weight_decay =0.005                  # Weight decay in optimizer
    group_list = ['night','day','23','0','8','9','10']   # Groups
    if_PGT = True                    # Ture: Partial group training (PGT), False: individual group training (IGT)

    # Switches for using auxiliary information (Need to modify)
    Tb_flag = True           
    hgt_flag = True
    time_flag = False
    in_layer, out_layer = get_layer_num(Tb_flag, hgt_flag, time_flag,)

    # set the device
    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the writer path.
    writer_dir = os.path.join(work_root,'Tensorboard',exe_name)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)

    # make the directory to save the model parameters
    model_save_root = os.path.join(work_root, 'model_save')
    best_save_dir = os.path.join(model_save_root, exe_name,)
    if not os.path.exists(best_save_dir):
        os.makedirs(best_save_dir)

    Mean_Model_err = 0
    Mean_Test_loss = 0
    Test_num = 0
    model = LTRUnet(in_layer, out_layer).to(DEVICE)

    for group_name in group_list:
        print('_______________________________________________________________________')
        print(group_name)
        best_save_location = os.path.join(best_save_dir, group_name+'.pt')

        # Set the writer
        writer_path = os.path.join(writer_dir,group_name)
        writer = SummaryWriter(log_dir=writer_path)

        # select samples for each group
        select = get_select_from_group(group_name)
        if group_name == 'night' or group_name == 'day':
            test_select = select
        else:
            test_select = ['_' + str(int(group_name[:2])).rjust(2,'0') + '-']

        Train_loader, Test_loader, Validation_loader = load_data(data_root, sample_list_root, select, test_select, 
                                                                 out_layer, Tb_flag,hgt_flag,  time_flag, BATCH_SIZE, num_workers, if_PGT)

        # Load the model
        model.load_state_dict(torch.load(basic_save_location,map_location='cuda:0'))

        # Train the model
        Train_step(model, DEVICE, EPOCHS, BATCH_SIZE,lr, weight_decay,best_save_location, writer, Train_loader, Validation_loader,if_PGT)
        
        #Test the model
        Test_step(model, DEVICE, best_save_location, Test_loader)

if __name__ == "__main__":
    main()