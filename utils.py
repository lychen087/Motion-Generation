import numpy as np
import pandas as pd
import copy, csv
import scipy.signal


def read_annot_csv(annot_file):

    with open(annot_file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        data_list = list(data)
        data_key = data_list[0]
        data_list = data_list[1:-1]
        
    return data_key, data_list


def remove_data_nan(data_arr, axis=0):

    data_arr = pd.DataFrame(data_arr)
    data_arr = data_arr.interpolate(axis= axis, limit_direction= 'both')
    data_arr = np.array(data_arr)

    return data_arr

### Training utils function
def remove_nan_3d_array(data_arr, axis=1):
    '''
    replace nan for 3d array
    input: data_arr [batch, time, feature]
    '''

    new_data_arr = np.zeros_like(data_arr)

    for batch_inx in range(data_arr.shape[0]):
        data_slice = data_arr[batch_inx, :, :]
        data_slice = pd.DataFrame(data_slice)
        data_slice = data_slice.interpolate(axis= int(axis-1), limit_direction= 'both')
        data_slice = np.array(data_slice)
        new_data_arr[batch_inx, :, :] = data_slice

    return new_data_arr

def replace_nan_by_mean_3d_array(data_arr):
    '''
    replace 3d data array by the data mean of feature (axis 2)
    input: data_arr [batch, time, feature]
    output: new_data_arr [batch, time, feature]
    '''

    mean_arr = np.nanmean(data_arr, axis=(0, 1))
    nan_arr = np.isnan(data_arr)
    nan_inx = np.where(nan_arr == True)
    # print('replace by mean value:', mean_arr)
    # print('nan inx:', nan_inx)
    
    for inx_0, inx_1, inx_2 in zip(nan_inx[0], nan_inx[1], nan_inx[2]):
        marker_mean = mean_arr[inx_2]
        data_arr[inx_0, inx_1, inx_2] = marker_mean
        
    return data_arr


def convert_str_to_class_label(
        str_label_arr,
        label,
        class_name):

    class_label_arr = np.zeros(str_label_arr.shape).astype(float)

    for inx in range(len(label)):
        item_inx = np.where(str_label_arr == label[inx])
        class_label_arr[item_inx] = class_name[inx]
    
    return class_label_arr


def clean_outlier_keep_dim(data_arr, tolerance = 1.5, keep_dim = 2):
    '''
    clean outliers for 3d input arr
    keep one dimension
    
    input: data_arr[clip, time, feature]
    
    output:clean_arr [clip, time, feature]
    
    for motion data: keep_dim = 2 (feature, xyz)
    tolerance: Q75/Q25  +- Quartile*tolerance
    '''

    all_dim = [0, 1, 2]
    clean_dim = tuple(set(all_dim) - set([keep_dim]))
    clean_arr = copy.deepcopy(data_arr)

    # get outlier upper/lower bound
    q75,q25 = np.percentile(data_arr,[75,25], axis=clean_dim)
    intr_qr = q75-q25
    upper_bound = q75+(tolerance*intr_qr) # [feature,]
    lower_bound = q25-(tolerance*intr_qr) # [feature,]

    # loop over feature dimension
    for f_inx in range(data_arr.shape[-1]):
        f_upper = upper_bound[f_inx] # value
        f_lower = lower_bound[f_inx] # value

        f_arr = data_arr[:, :, f_inx]
        f_arr = np.where(f_arr > f_upper, f_upper, f_arr)
        f_arr = np.where(f_arr < f_lower, f_lower, f_arr)
        clean_arr[:, :, f_inx] = f_arr
        clean_arr = np.array(clean_arr)
    
    return clean_arr


def normalize_data_all_dim(data_arr, norm_max = 1):
    '''
    normalize 3D input arr to the range [0, norm_max]
    
    input: data_arr[batch, time, feature]
    
    output:norm_data [batch, time, feature]
    data_min
    data_max
    
    '''
                                                    
    data_min = np.min(data_arr)
    data_max = np.max(data_arr)
    
    norm_data = np.array((data_arr - data_min) * norm_max / (data_max - data_min))
    
    return norm_data, data_min, data_max


def convert_str_to_one_hot_label(
        str_label_arr,
        label,
        one_hot):
    '''
    convert str label array to one hot array
    input: str_label_arr [batch, time]
    label: list of string label, 
    e.g., ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff']
    ['legato', 'staccato']
    one_hot: correspond value of label
    e.g., [0,0,1,2,3,4,5,5], [0,1]

    output: one_hot_label_arr [batch, time , n_class]
    '''

    int_label_arr = np.zeros((str_label_arr.shape[0], str_label_arr.shape[1]))

    for name, value in zip(label, one_hot):
        # print(name, value)
        int_label_arr[str_label_arr == name] = value
    # print(int_label_arr[:10, :5])

    one_hot_arr = (np.arange(np.array(one_hot).max() + 1) == int_label_arr[...,None]).astype(float)
    # print(one_hot_arr[:10, :5, :])
    print('convert label array to one hot array')
    print('label array:', int_label_arr.shape)
    print('one hot array:', one_hot_arr.shape)

    return one_hot_arr

def split_train_eval_sub_pkg(data_pkg, train_eval_split_ratio):
    '''
    split data_pkg to train_pkg & eval_pkg
    input: data_pkg list {key: data_arr}
    output: train_pkg list {key: data_arr},
            eval_pkg list {key: data_arr}
    '''

    train_pkg = [] 
    eval_pkg = []

    for inx in range(len(data_pkg)):

        if inx % train_eval_split_ratio == 0:
            eval_pkg.append(data_pkg[inx])
            # print(inx, 'assign to eval pkg')
        
        elif inx % train_eval_split_ratio != 0:
            train_pkg.append(data_pkg[inx])
            # print(inx, 'assign to train pkg')
    
    print('train pkg:', len(train_pkg))
    print('eval pkg:', len(eval_pkg))

    del data_pkg
    return train_pkg, eval_pkg


###


def shift_pos_to_joint_mean(motion_arr):
    '''
    shift each joint to its own mean position
    get the mean pos of each joint's xyz
    input: 3d motion_arr [batch, time, feature (joint*xyz)]
    output: shifted joint arr pos_shift [batch, time, feature (joint*xyz)]
    the mean position of each joint pos_mean [feature (joint*xyz)]
    '''

    pos_mean = np.mean(motion_arr, axis= (0, 1))
    pos_shift = np.array(motion_arr - pos_mean)

    return pos_shift, pos_mean

def resample_2d_array(
        data_arr, 
        original_sample_rate, 
        final_sample_rate
    ):
    
    '''
    resample time (axis 0) for the ratio for the resample rate
    input: 2d data array [time, feature]
    output: resample 2d array [resample_time, feature]
    resample_time = time * final_sample_rate /original_sample_rate
    '''

    resample_num = int(data_arr.shape[0] * final_sample_rate / original_sample_rate)
    resample_arr = np.zeros((resample_num, data_arr.shape[1]))

    original_time = np.linspace(0, data_arr.shape[0], data_arr.shape[0])
    resample_time = np.linspace(0, data_arr.shape[0], resample_num)

    for marker_inx in range(data_arr.shape[1]):
        data_slice = data_arr[:, marker_inx]
        interp_fn = scipy.interpolate.interp1d(original_time, data_slice)
        resample_slice = interp_fn(resample_time)
        resample_arr[:, marker_inx] = resample_slice

    return resample_arr

# get joint center position of the input motion segment data
def get_joint_center(seg_data):

    seg_x = np.mean(seg_data[:, ::3], axis=1)
    seg_y = np.mean(seg_data[:, 1::3], axis=1)
    seg_z = np.mean(seg_data[:, 2::3], axis=1)
    
    seg_center = np.array([seg_x, seg_y, seg_z]).T
    
    return seg_center


# translate motion_arr to local coordinate with ref_pos is (0, 0, 0) 
def get_local_coordinate(motion_arr, ref_pos):

    new_arr = np.zeros(motion_arr.shape)
    motion_arr = np.array(motion_arr)

    pos_x = motion_arr[:, ::3]
    pos_y = motion_arr[:, 1::3]
    pos_z = motion_arr[:, 2::3]

    trans_x = pos_x - ref_pos[0]
    trans_y = pos_y - ref_pos[1]
    trans_z = pos_z - ref_pos[2]

    new_arr[:, ::3] = trans_x
    new_arr[:, 1::3] = trans_y
    new_arr[:, 2::3] = trans_z
    
    return new_arr


def remove_extreme_values(data_arr, threshold = (-100, 100)):
    '''
    remove extreme values
    input: data array
    threshold: (lower_bound, upper_bound)
    '''
    
    clean_arr = np.where(data_arr > threshold[1], threshold[1], data_arr)
    clean_arr = np.where(clean_arr < threshold[0], threshold[0], clean_arr)
    
    # count removed values
    remove_small = np.count_nonzero(data_arr < threshold[0])
    remove_large = np.count_nonzero(data_arr > threshold[1])
    # print('remove extreme large values:', remove_large, 'remove extreme small values:', remove_small)
    
    return clean_arr