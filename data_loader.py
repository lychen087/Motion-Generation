from torch.utils.data import DataLoader, Dataset
import torch

class MotionDataset(Dataset):
    def __init__(self, data):
        self.audio_data = torch.tensor(data["audio_data"], dtype=torch.float32)
        self.feature_data = torch.tensor(data["comb_label"], dtype=torch.float32)
        self.motion_head_data = torch.tensor(data["motion_pos_head"], dtype=torch.float32)
        self.motion_rhand_data = torch.tensor(data["motion_pos_rhand"], dtype=torch.float32)
        self.motion_lhand_data = torch.tensor(data["motion_pos_lhand"], dtype=torch.float32)
        self.motion_torso_data = torch.tensor(data["motion_pos_torso"], dtype=torch.float32)
        self.motion_root_data = torch.tensor(data["motion_pos_root"], dtype=torch.float32)
        self.motion_bridge_data = torch.tensor(data["motion_pos_bridge"], dtype=torch.float32)
        self.motion_volute_data = torch.tensor(data["motion_pos_volute"], dtype=torch.float32)

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return {
            "audio_data": self.audio_data[idx],
            "feature_data": self.feature_data[idx],
            "motion_head_data": self.motion_head_data[idx],
            "motion_rhand_data": self.motion_rhand_data[idx],
            "motion_lhand_data": self.motion_lhand_data[idx],
            "motion_torso_data": self.motion_torso_data[idx],
            "motion_root_data": self.motion_root_data[idx],
            "motion_bridge_data": self.motion_bridge_data[idx],
            "motion_volute_data": self.motion_volute_data[idx]
        }
