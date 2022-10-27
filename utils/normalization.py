import numpy as np

input_mean = 295.2741566794259 
input_std = 2.2389705717386232

output_mean = 292.9008821941557 
output_std = 2.6616024114014136

def normalize(put_data, kind):
    """ Normalize the input or the label """
    if kind == 'input':
        if put_data.shape[0]>1: 
            # input include T2 and Tempreture background
            put_data[0] = (put_data[0]-input_mean)/input_std
            put_data[1:] = (put_data[1:]-output_mean)/output_std
            return put_data
        else:
            # input only include T2
            return (put_data-input_mean)/input_std
    if kind == 'label':
        return (put_data-output_mean)/output_std

def turn_back_label(put_data):
    """ Restore dimensionless label to the original scale """
    return (put_data*output_std+output_mean)

def turn_back_T2(put_data):
    """ Restore dimensionless intput(T2) to the original scale """
    return (put_data*input_std+input_mean)