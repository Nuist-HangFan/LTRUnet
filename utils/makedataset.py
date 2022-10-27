import os
import numpy as np
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as DDataset
import h5py


# Read the terrain height data
HGT_file_path = '~/data/SZ/Raw_T_data/202006_00_T/2020-06-18_21-00-00'
# HGT_file = DDataset(HGT_file_path)

HGT_file = DDataset('/public/home/FanHang/data/SZ/Raw_T_data/202006_00_T/2020-06-18_21-00-00')
HGT = HGT_file.variables['HGT'][0][:224,:224].data
HGT_mean = np.mean(HGT)
HGT_std = np.std(HGT)
HGT = (HGT-HGT_mean)/HGT_std
HGT = HGT[np.newaxis,:]

class Make_Dataset(Dataset):
    def __init__(self, data_list, data_path, out_layer, Tb_flag = False, hgt_flag=False, time_flag=False, TB_only_flag=False):
        """ Prepare the Dataset for training, testing or validation 
        Args:
            data_list: the list of all file names for dataset.
            data_path: path of the original data
            in_layer: the number of the input channels.
            out_layer: the number of the output channels.
            hgt_flag: switch for using terrain in input
            time_flag: switch for using time infromation in input
            TB_only_flag: switch for using temperature background as the only input of the model.
        Returns:
            list: list of input data and label data
        """
        self.path = data_path
        self.data_list = data_list
        self.out_layer = out_layer
        self.hgt_flag = hgt_flag
        self.HGT = HGT
        self.time_flag = time_flag
        self.TB_only_flag = TB_only_flag
        self.Tb_flag = Tb_flag

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        f = h5py.File(os.path.join(self.path, data_name),'r') 
        if self.Tb_flag:
            input_data = f['put_in'][:14]
        else:
            input_data = f['put_in'][:1]
        label_data = f['label'][:self.out_layer]
        f.close()   

        if self.time_flag:
            hour_layer = np.ones((1,224,224))*int(data_name[11:13])*0.04166
            hour_layer = hour_layer.astype('float32')
            input_data = np.concatenate((input_data, hour_layer), 0)

        if self.hgt_flag:
            input_data = np.concatenate((input_data, self.HGT), 0)
        if self.TB_only_flag:
            f = h5py.File(os.path.join(self.path, data_name),'r') 
            input_data = f['put_in'][1:14]
            label_data = f['label'][:self.out_layer]
            f.close()

        return input_data, label_data


def read_txt(path):
    """ Read the txt by line and retrn a list """
    file = open(path, "r")
    data_list = []
    while True:
        line = file.readline()
        if line:
            data_list.append(line[:-5])
        else:
            break
    file.close()
    return data_list

def select_data(data_list, select):
    """ Select files whose names contain specific strings.
    Args:
        data_list: the original data list
        select: a list of specific strings
    Return:
        list: list of data names contained specific strings.
    """
    if select:
        final_list = []
        for k in select:
            final_list += [i for i in data_list if k in i]
        data_list = final_list
    return data_list

def Loader(dataset, BATCH_SIZE, num_workers, shuffle=True):
    """ Return a data loader """
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

