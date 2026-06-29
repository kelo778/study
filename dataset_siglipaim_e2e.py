
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils3_e2e import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d

# 用于早期融合
def get_data_dict(feature_dir_RGB, feature_dir_FLOW, label_dir, video_list, event_list, sample_rate=4, temporal_aug=True, boundary_smooth=None,proposal_dir=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file_RGB = os.path.join(feature_dir_RGB, '{}.npy'.format(video))
        feature_file_FLOW = os.path.join(feature_dir_FLOW, '{}.npy'.format(video))
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)
        
                # ===== Load EAST proposal probability =====
        if proposal_dir is not None:
            proposal_file = os.path.join(proposal_dir, f'{video}.npz')
            if not os.path.exists(proposal_file):
                raise FileNotFoundError(f'EAST proposal not found: {proposal_file}')

            npz = np.load(proposal_file, allow_pickle=True)
            if 'prob' not in npz.files:
                raise KeyError(f'Cannot find "prob" in {proposal_file}. Available keys: {npz.files}')

            proposal_prob_raw = npz['prob'].astype(np.float32)

            # EAST 通常保存为 [T, C]，这里统一转成 [C, T]
            if proposal_prob_raw.ndim != 2:
                raise ValueError(f'Invalid prob shape in {proposal_file}: {proposal_prob_raw.shape}')

            if proposal_prob_raw.shape[1] == len(event_list):
                proposal_prob_raw = proposal_prob_raw.T  # [T, C] -> [C, T]
            elif proposal_prob_raw.shape[0] == len(event_list):
                pass  # already [C, T]
            else:
                raise ValueError(
                    f'Cannot infer class dimension for {proposal_file}, '
                    f'prob shape={proposal_prob_raw.shape}, num_classes={len(event_list)}'
                )

            # 归一化，防止不是严格概率
            proposal_prob_raw = np.maximum(proposal_prob_raw, 1e-8)
            proposal_prob_raw = proposal_prob_raw / proposal_prob_raw.sum(axis=0, keepdims=True)

            # 先强制检查长度。如果这里报错，说明 EAST proposal 的 T 和 groundTruth 帧数不一致
            if proposal_prob_raw.shape[1] != frame_num:
                raise ValueError(
                    f'Length mismatch in {video}: '
                    f'GT length={frame_num}, proposal length={proposal_prob_raw.shape[1]}'
                )
        else:
            # 没有 proposal 时，
            raise ValueError("proposal_dir is None. EAST proposal is required for this e2e experiment.")
        
        feature_rgb = np.load(feature_file_RGB, allow_pickle=True)
        feature_flow = np.load(feature_file_FLOW, allow_pickle=True)

        # --- [修正点]：自动检测并处理原始特征可能是 [Dim, Time] 或 [Time, Dim] 的情况 ---
        # 目标：确保 feature 为[Crops, Time, Dim] 格式，以便后续按 Time 维度进行采样
        def adjust_shape(feat, expected_dim, f_num):
            if len(feat.shape) == 2:
                # 判断是否为 [Dim, Time]
                if feat.shape[0] == expected_dim and feat.shape[1] != expected_dim:
                    feat = feat.T  # 转置为 [Time, Dim]
                elif feat.shape[0] != expected_dim and feat.shape[1] == expected_dim:
                    pass  # 已经是 [Time, Dim]，无需转置
                else:
                    # 如果维度有些许偏差，依靠哪一维更接近帧数来判断 Time 维
                    if abs(feat.shape[1] - f_num) < abs(feat.shape[0] - f_num):
                        feat = feat.T
                feat = np.expand_dims(feat, 0) # 统一变成 [1, Time, Dim]
                
            elif len(feat.shape) == 3:
                # 判断是否为 [Crops, Dim, Time]
                if feat.shape[1] == expected_dim and feat.shape[2] != expected_dim:
                    feat = np.transpose(feat, (0, 2, 1)) # 转置为 [Crops, Time, Dim]
                elif feat.shape[1] != expected_dim and feat.shape[2] == expected_dim:
                    pass # 已经是 [Crops, Time, Dim]
                else:
                    if abs(feat.shape[2] - f_num) < abs(feat.shape[1] - f_num):
                        feat = np.transpose(feat, (0, 2, 1))
            else:
                raise Exception(f'Invalid Feature Shape: {feat.shape}')
            return feat

        # 假设 RGB 特征维度为 1152，Flow 特征维度为 1024
        feature_rgb = adjust_shape(feature_rgb, expected_dim=1152, f_num=frame_num)
        feature_flow = adjust_shape(feature_flow, expected_dim=1024, f_num=frame_num)
        # ==============================================================================

        # 此时 feature_rgb 形状一定为 [Crops, Time, 1152]
        # 此时 feature_flow 形状一定为 [Crops, Time, 1024]
                                
        if temporal_aug:
            # 在 Time 维度 (axis=1) 上进行采样
            feature_rgb =[
                feature_rgb[:, offset::sample_rate, :]
                for offset in range(sample_rate)
            ]

            feature_flow = [
                feature_flow[:, offset::sample_rate, :]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext =[
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
            
            proposal_prob_ext = [
                proposal_prob_raw[:, offset::sample_rate]
                for offset in range(sample_rate)
            ]
        else:
            feature_rgb = [feature_rgb[:,::sample_rate,:]]
            feature_flow =[feature_flow[:,::sample_rate,:]]  
            event_seq_ext = [event_seq_raw[::sample_rate]]
            boundary_seq_ext = [boundary_seq_raw[::sample_rate]]
            proposal_prob_ext = [proposal_prob_raw[:, ::sample_rate]]
            
        data_dict[video]['feature_rgb'] =[torch.from_numpy(i).float() for i in feature_rgb]
        data_dict[video]['feature_flow'] =[torch.from_numpy(i).float() for i in feature_flow]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['event_seq_ext'] =[torch.from_numpy(i).float() for i in event_seq_ext]
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()
        data_dict[video]['boundary_seq_ext'] = [torch.from_numpy(i).float() for i in boundary_seq_ext]
        data_dict[video]['proposal_prob_raw'] = torch.from_numpy(proposal_prob_raw).float()
        data_dict[video]['proposal_prob_ext'] = [
            torch.from_numpy(i).float() for i in proposal_prob_ext
        ]
    return data_dict

def get_boundary_seq(event_seq, boundary_smooth=None):
    boundary_seq = np.zeros_like(event_seq)

    _, start_times, end_times = get_labels_start_end_time([str(int(i)) for i in event_seq])
    boundaries = start_times[1:]
    if len(boundaries) > 0: # 增加安全性检查
        assert min(boundaries) > 0
        boundary_seq[boundaries] = 1
        boundary_seq[[i-1 for i in boundaries]] = 1

    if boundary_smooth is not None:
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
        
        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        if temp_seq.shape[0] > 1:
            temp_seq[temp_seq.shape[0] // 2] = 1
            temp_seq[temp_seq.shape[0] // 2 - 1] = 1
            norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
            if norm_z > 0:
                boundary_seq[boundary_seq > norm_z] = norm_z
                boundary_seq /= boundary_seq.max()

    return boundary_seq


def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out


class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, class_num, mode):
        super(VideoFeatureDataset, self).__init__()
        assert(mode in ['train', 'test'])
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list =[i for i in self.data_dict.keys()]
        
    def get_class_weights(self):
        full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)
        return class_weights

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature_rgb = self.data_dict[video]['feature_rgb']
            feature_flow = self.data_dict[video]['feature_flow']
            label = self.data_dict[video]['event_seq_ext']
            boundary = self.data_dict[video]['boundary_seq_ext']
            proposal_prob = self.data_dict[video]['proposal_prob_ext']
            
            temporal_aug_num = len(feature_flow)
            temporal_rid = random.randint(0, temporal_aug_num - 1)
            feature_rgb = feature_rgb[temporal_rid]
            feature_flow = feature_flow[temporal_rid]
            label = label[temporal_rid]
            boundary = boundary[temporal_rid]
            proposal_prob = proposal_prob[temporal_rid]

            spatial_aug_num = feature_flow.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1)
            
            # 取出一个 crop
            feature_rgb = feature_rgb[spatial_rid]   # 形状此时一定为[Time, Dim]
            feature_flow = feature_flow[spatial_rid] # 形状此时一定为[Time, Dim]
            
            # --- 关键：转置为 [Dim, Time] 以适配 Conv1d ---
            feature_rgb = feature_rgb.T   # 变成 [Dim, Time] (即 [1152, T])
            feature_flow = feature_flow.T # 变成[Dim, Time] (即 [1024, T])
        
            boundary = boundary.unsqueeze(0)
            boundary /= boundary.max() if boundary.max() > 0 else 1.0 
            
        if self.mode == 'test':

            feature_rgb = self.data_dict[video]['feature_rgb']
            feature_flow = self.data_dict[video]['feature_flow']
            label = self.data_dict[video]['event_seq_raw']
            boundary = self.data_dict[video]['boundary_seq_ext']
            proposal_prob = self.data_dict[video]['proposal_prob_ext']
            
            # Test 模式下，输入目前是[Crops, Time, Dim]
            # 我们需要让它变成 [Crops, Dim, Time]
            feature_rgb = [torch.swapaxes(i, 1, 2) for i in feature_rgb]
            feature_flow =[torch.swapaxes(i, 1, 2) for i in feature_flow]
            
            label = label.unsqueeze(0)   # 1 X T'  
            boundary = [i.unsqueeze(0).unsqueeze(0) for i in boundary]   #[1 x 1 x T]  
            proposal_prob = [i.unsqueeze(0) for i in proposal_prob]  # each: [1, C, T]
        return feature_rgb, feature_flow, label, boundary, proposal_prob, video