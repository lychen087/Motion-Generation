import os, random, joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchviz import make_dot
from model import MotionGenerationTransformer
from utils import *
from config import config
from data_loader import MotionDataset

# MOTION_NORMALIZE_RANGE = config["input_data"]["motion_normalize_range"]

NUM_EPOCHS = config["training"]["epoch_num"]
BATCH_SIZE = config["training"]["batch_size"]
VALIDATION_SPLIT = config["training"]["validation_split"]

def get_body_seg_data_from_joint_arr(joint_arr):

    # rhand = joint_arr[:, :, 12:21] 
    # lhand = joint_arr[:, :, 24:33]
    # pos_rhand = np.concatenate((btop, rhand), axis=-1)
    # pos_lhand = np.concatenate((vtop, lhand), axis=-1)
    btop = joint_arr[:, :, 57:60]
    bbom = joint_arr[:, :, 60:63]
    vtop = joint_arr[:, :, 51:54]
    vbom = joint_arr[:, :, 54:57]

    pos_head = np.array(joint_arr[:, :, :6])
    pos_rhand = np.array(joint_arr[:, :, 12:21]) # RShoulder, RElbow, RWrist
    pos_lhand = np.array(joint_arr[:, :, 24:33])
    pos_torso = np.array(joint_arr[:, :, 63:66])
    pos_root = np.array(joint_arr[:, :, 6:9])
    pos_bridge = np.concatenate((btop, bbom), axis=-1)
    pos_volute = np.concatenate((vtop, vbom), axis=-1)
  

    return pos_head, pos_rhand, pos_lhand, pos_torso, pos_root, pos_bridge, pos_volute


def get_motion_generation_pkg(data_pkg, norm_range = 1):

    new_data_pkg = {}

    # get audio data
    new_data_pkg['audio_data'] = data_pkg['audio_data']

    # combine multiple labels
    not_num_arr = np.ones(data_pkg['beat_label'].shape)
    comb_label = np.concatenate((
        not_num_arr, 
        data_pkg['beat_label'],
        data_pkg['downbeat_label'],
        data_pkg['phrase_label'],
        data_pkg['dyn_label'],
        data_pkg['arti_label'],
        data_pkg['midi_label'],
        data_pkg['midi_label'],
        data_pkg['flux'],
        data_pkg['rms']), axis = -1
    )
    
    new_data_pkg['comb_label'] = comb_label

    # get body segment data
    pos_head_data, pos_rhand_data, pos_lhand_data, pos_torso_data, pos_root_data, pos_bridge_data, pos_volute_data \
        = get_body_seg_data_from_joint_arr(data_pkg['motion_pos_data'])
    
    new_data_pkg['motion_pos_head'] = pos_head_data
    new_data_pkg['motion_pos_rhand'] = pos_rhand_data
    new_data_pkg['motion_pos_lhand'] = pos_lhand_data
    new_data_pkg['motion_pos_root'] = pos_torso_data
    new_data_pkg['motion_pos_torso'] = pos_root_data
    new_data_pkg['motion_pos_bridge'] = pos_bridge_data
    new_data_pkg['motion_pos_volute'] = pos_volute_data

    # normalize motion
    motion_key_list = ['motion_pos_head', 'motion_pos_rhand', 'motion_pos_lhand', 'motion_pos_root', 'motion_pos_bridge', 'motion_pos_volute']
    mean_key_list = ['mean_pos_head', 'mean_pos_rhand', 'mean_pos_lhand', 'mean_pos_root', 'mean_pos_bridge', 'mean_pos_volute']
    min_max_key_list = ['min_max_head', 'min_max_rhand', 'min_max_lhand', 'min_max_root', 'min_max_bridge', 'min_max_volute']

    for inx in range(len(motion_key_list)):
        # shift to the pos where joint mean is 0
        motion_key = motion_key_list[inx]
        new_data_pkg[motion_key], new_data_pkg[mean_key_list[inx]] = shift_pos_to_joint_mean(new_data_pkg[motion_key])
        new_data_pkg[motion_key] = clean_outlier_keep_dim(
            new_data_pkg[motion_key], 
            tolerance = 1.5, keep_dim = 2
        )
        
        # normalize the range of motion data
        new_data_pkg[motion_key], data_min, data_max = normalize_data_all_dim(
            new_data_pkg[motion_key], norm_max = norm_range
        )
        new_data_pkg[min_max_key_list[inx]] = np.array([data_min, data_max])
    

    # check data
    print('---')  
    print('Normalized data:')
    for key in new_data_pkg.keys():
        print(key, new_data_pkg[key].shape)
        # print('data type:', load_data_pkg[key].dtype)

        if key == 'motion_pos_head' or key == 'motion_pos_rhand' or key == 'motion_pos_lhand' \
            or key == 'motion_pos_root' or key == 'motion_pos_bridge' or key == 'motion_pos_volute':
            print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(new_data_pkg[key]), 
                np.nanmin(new_data_pkg[key]),
                np.nanmean(new_data_pkg[key])))

    del data_pkg
    return new_data_pkg

def get_data_and_label_from_pkg(data_pkg):
    '''
    reorganize data and label as array for input data package
    input: data_pkg: list {keys: array}
    output: data_pkg dict {keys, array}
    '''

    # shuffle data & label
    data_pkg = random.sample(data_pkg, len(data_pkg))

    # get data & label array
    load_data_pkg = {}

    for key in data_pkg[0].keys(): 

        try:
            data_arr = np.array([item[key] for item in data_pkg], dtype=np.float32)
        
        except: 
            data_arr = np.array([item[key] for item in data_pkg])
        
        # clean up data
        if key == 'audio_data' or key == 'motion_pos_data' or key == 'motion_vel_data':
            new_data_arr = remove_nan_3d_array(data_arr, axis=1)
            new_data_arr = replace_nan_by_mean_3d_array(data_arr)
            # print(key, 'processing audio, motion data')
            
        # convert string label to class label
        elif key == 'dyn_label':

            new_data_arr = convert_str_to_class_label(
                data_arr,
                label = ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'],
                class_name = [0,0,1,2,3,4,5,5]
            )
            new_data_arr = new_data_arr[..., np.newaxis]
            # print(key, 'processing dyn label')
        
        elif key == 'arti_label':

            new_data_arr = convert_str_to_class_label(
                data_arr,
                label = ['legato', 'none', 'staccato'],
                class_name = [0,1,2]
            )
            new_data_arr = new_data_arr[..., np.newaxis]
            # print(key, 'processing arti label')
        
        elif key == 'beat_label' or key == 'downbeat_label' or key== 'phrase_label':
            new_data_arr = data_arr[..., np.newaxis]
            # print(key, 'processing beat, downbeat, phrase labels')
        
        else:
            new_data_arr = data_arr
            # print(key, 'maintain original data')
        
        load_data_pkg[key] = new_data_arr

    # check data values
    print('---') 
    print('data values:')
    for key in load_data_pkg.keys():
        print(key, load_data_pkg[key].shape)
        # print('data type:', load_data_pkg[key].dtype)

        if key == 'audio_data' or key == 'motion_pos_data' or key == 'motion_vel_data':
            print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(load_data_pkg[key]), 
                np.nanmin(load_data_pkg[key]),
                np.nanmean(load_data_pkg[key])
            ))
        
    del data_pkg
    return load_data_pkg
    


def load_data_pkg_fn(data_pkl_file, 
                     train_eval_split_ratio = 10, 
                     norm_range = 1):

    input_data_pkg = joblib.load(data_pkl_file)
    print('Load data keys:', input_data_pkg[0].keys())
    print('Load data len:', len(input_data_pkg))

    train_data_pkg, eval_data_pkg = split_train_eval_sub_pkg(
        input_data_pkg, 
        train_eval_split_ratio = train_eval_split_ratio
    )   
    
    del input_data_pkg

    print('=====')
    print('get train data:')
    train_data = get_data_and_label_from_pkg(train_data_pkg)
    train_data = get_motion_generation_pkg(train_data, norm_range = norm_range)

    print('=====')
    print('get eval data:')
    eval_data = get_data_and_label_from_pkg(eval_data_pkg)
    eval_data = get_motion_generation_pkg(eval_data, norm_range = norm_range)

    del train_data_pkg
    del eval_data_pkg
    
    return train_data, eval_data




def training(model, train_data, eval_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            audio_input, feature_input = batch["audio_data"], batch["feature_data"]
            rhand_labels, lhand_labels = batch["motion_rhand_data"], batch["motion_lhand_data"]
            head_labels, torso_labels, root_labels = batch["motion_head_data"], batch["motion_torso_data"], batch["motion_root_data"]
            bridge_labels, volute_labels = batch["motion_bridge_data"], batch["motion_volute_data"]

            # Forward pass
            rhand_pred, lhand_pred, head_pred, torso_pred, root_pred, bridge_pred, volute_pred = model(audio_input, feature_input)
            
            loss = (
                criterion(rhand_pred, rhand_labels) + 
                criterion(lhand_pred, lhand_labels) + 
                criterion(head_pred, head_labels) +
                criterion(torso_pred, torso_labels) + 
                criterion(root_pred, root_labels) + 
                criterion(bridge_pred, bridge_labels) +
                criterion(volute_pred, volute_labels)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}")


if __name__ == '__main__':

    # define input data directory
    instrument_name = 'violin' 
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(script_folder, os.pardir))
    input_data_pkl_file = os.path.join(script_folder, instrument_name + "_training_data_pkg.pkl")
    print('Input data from:', input_data_pkl_file)
    
    # define save model directory
    save_model_dir = script_folder
    print('Define save model folder:', save_model_dir)


    # ===== load data =====
    train_data, eval_data = load_data_pkg_fn(
        input_data_pkl_file, 
        train_eval_split_ratio = int(1/VALIDATION_SPLIT),
        norm_range = 50, # MOTION_NORMALIZE_RANGE
    )
    print('=====\n')

    model = MotionGenerationTransformer(
        audio_feature_dim=128, 
        label_feature_dim=10, 
        time_frame_size=640, 
        audio_embed_dim=64,
        label_embed_dim=16,
        ff_dim=1024, 
        hidden_dim=256, 
        num_heads=4, 
        num_layers_main=3, 
        num_layers_branch=2, 
        output_dims={
            "head": 6, 
            "rhand": 9, 
            "lhand": 9, 
            "torso": 3, 
            "root": 3,
            "bridge": 6,
            "volute": 6
        }
    )

    train_dataset = MotionDataset(train_data)
    eval_dataset = MotionDataset(eval_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # training(model, train_data, eval_data)



    audio_input = torch.randn(1, 640, 128)  # [batch_size, time_frame_size, audio_feature_dim]
    feature_input = torch.randn(1, 640, 10)  # [batch_size, time_frame_size, label_feature_dim]


    output = model(audio_input, feature_input)


    dot = make_dot(output, params=dict(list(model.named_parameters())))
    dot.format = "png"  
    dot.render("motion_generation_transformer")  