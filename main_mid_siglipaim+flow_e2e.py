import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset_siglipaim_e2e import restore_full_sequence
from dataset_siglipaim_e2e import get_data_dict
from dataset_siglipaim_e2e import VideoFeatureDataset
from fvcore.nn import FlopCountAnalysis
from model_cross_attention_mask2_e2e import ASDiffusionModel
#from model_cross_attention_50salads import ASDiffusionModel   50salads用这个
from tqdm import tqdm
from utils3_e2e import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils3_e2e import mode_filter


class Trainer:
    def __init__(self, encoder_params_rgb, encoder_params_flow, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
#         self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params_rgb, encoder_params_flow, decoder_params, diffusion_params, self.num_classes, self.device)
        
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                feature_rgb, feature_flow, label, boundary, proposal_prob, video = data
                
                # 放入 GPU
                feature_rgb = feature_rgb.to(device)
                feature_flow = feature_flow.to(device)
                label = label.to(device)
                boundary = boundary.to(device)
                proposal_prob = proposal_prob.to(device)

                # ================= 核心修改开始 =================
                # 检查并修复维度：从 (Batch, Time, Channel) -> (Batch, Channel, Time)
                # SigLIP 维度是 1152
                if feature_rgb.ndim == 3 and feature_rgb.shape[2] == 1152:
                    feature_rgb = feature_rgb.permute(0, 2, 1)
                
                # Flow 维度是 1024
                if feature_flow.ndim == 3 and feature_flow.shape[2] == 1024:
                    feature_flow = feature_flow.permute(0, 2, 1)

                # +++ 新增：对齐时间维度以防止拼接崩溃 +++
                # 取 label 的时间维度作为目标长度 (Batch_size, Time)
                target_len = label.shape[-1]
                if feature_rgb.shape[-1] != target_len:
                    feature_rgb = F.interpolate(feature_rgb, size=target_len, mode='linear', align_corners=False)
                if feature_flow.shape[-1] != target_len:
                    feature_flow = F.interpolate(feature_flow, size=target_len, mode='linear', align_corners=False)
                if proposal_prob.shape[-1] != target_len:
                    proposal_prob = F.interpolate(proposal_prob, size=target_len, mode='nearest')
                # ================= 核心修改结束 =================

                loss_dict = self.model.get_training_loss(feature_rgb, feature_flow, 
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    proposal_prob=proposal_prob,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # for mode in['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k,v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k,v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k,v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k,v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in['median', 'mode', 'purge', None])


        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature_rgb_list, feature_flow_list, label, _, proposal_prob_list, video = test_dataset[video_idx]
            
            # test_dataset 返回的是 list of tensors，需要逐个处理
            
            if mode == 'encoder':
                # Encoder 模式需要适配 (C, T) 输入
                output =[]
                for i in range(len(feature_rgb_list)):
                    f_rgb = feature_rgb_list[i].to(device)
                    f_flow = feature_flow_list[i].to(device)
                    # 确保维度 (1, C, T)
                    if f_rgb.ndim == 2: f_rgb = f_rgb.unsqueeze(0)
                    if f_flow.ndim == 2: f_flow = f_flow.unsqueeze(0)
                    # 转置检查
                    if f_rgb.shape[1] != 1152 and f_rgb.shape[2] == 1152: f_rgb = f_rgb.permute(0, 2, 1)
                    if f_flow.shape[1] != 1024 and f_flow.shape[2] == 1024: f_flow = f_flow.permute(0, 2, 1)

                    # +++ 新增：推理阶段对齐时间维度 +++
                    target_len = f_rgb.shape[-1]
                    if f_flow.shape[-1] != target_len:
                        f_flow = F.interpolate(f_flow, size=target_len, mode='linear', align_corners=False)

                    # 这里假设 encoder 只用了 rgb，具体看你代码逻辑，如果是双流可能需要修改 encoder 调用
                    out = self.model.encoder_rgb(f_rgb) 
                    output.append(out)

                output =[F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output =[]
                for i in range(len(feature_rgb_list)):
                    f_rgb = feature_rgb_list[i].to(device)
                    f_flow = feature_flow_list[i].to(device)
                    
                    # 维度适配: (1, C, T)
                    if f_rgb.ndim == 2: f_rgb = f_rgb.unsqueeze(0)
                    if f_flow.ndim == 2: f_flow = f_flow.unsqueeze(0)
                    
                    # 维度转置检查: (Batch, Time, Channel) -> (Batch, Channel, Time)
                    if f_rgb.shape[1] != 1152 and f_rgb.shape[2] == 1152:
                        f_rgb = f_rgb.permute(0, 2, 1)
                    if f_flow.shape[1] != 1024 and f_flow.shape[2] == 1024:
                        f_flow = f_flow.permute(0, 2, 1)
                    
                    # +++ 新增：推理阶段对齐时间维度 +++
                    target_len = f_rgb.shape[-1]
                    if f_flow.shape[-1] != target_len:
                        f_flow = F.interpolate(f_flow, size=target_len, mode='linear', align_corners=False)

                    p = proposal_prob_list[i].to(device)
                    if p.ndim == 2:
                        p = p.unsqueeze(0)

                    if p.shape[-1] != f_rgb.shape[-1]:
                        p = F.interpolate(p, size=f_rgb.shape[-1], mode='nearest')

                    res = self.model.ddim_sample(f_rgb, f_flow, proposal_prob=p, seed=seed)
                    
                    output.append(res)
                    
                output =[i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                # 取中间的一个 crop
                mid_idx = len(feature_rgb_list) // 2
                f_rgb = feature_rgb_list[mid_idx].to(device)
                f_flow = feature_flow_list[mid_idx].to(device)
                
                if f_rgb.ndim == 2: f_rgb = f_rgb.unsqueeze(0)
                if f_flow.ndim == 2: f_flow = f_flow.unsqueeze(0)
                
                if f_rgb.shape[1] != 1152 and f_rgb.shape[2] == 1152: f_rgb = f_rgb.permute(0, 2, 1)
                if f_flow.shape[1] != 1024 and f_flow.shape[2] == 1024: f_flow = f_flow.permute(0, 2, 1)

                # +++ 新增：推理阶段对齐时间维度 +++
                target_len = f_rgb.shape[-1]
                if f_flow.shape[-1] != target_len:
                    f_flow = F.interpolate(f_flow, size=target_len, mode='linear', align_corners=False)
                
                p = proposal_prob_list[mid_idx].to(device)
                if p.ndim == 2:
                    p = p.unsqueeze(0)

                if p.shape[-1] != f_rgb.shape[-1]:
                    p = F.interpolate(p, size=f_rgb.shape[-1], mode='nearest')

                output = [self.model.ddim_sample(f_rgb, f_flow, proposal_prob=p, seed=seed)]
                output =[i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0
            
            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output =[i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            # ==============================================================
            # +++ 新增：修复 min_len 截断导致插值恢复时发生的 ValueError 错误 +++
            expected_len = len(np.arange(left_offset, label.shape[-1] - right_offset, self.sample_rate))
            if len(output) < expected_len:
                output = np.pad(output, (0, expected_len - len(output)), mode='edge')
            elif len(output) > expected_len:
                output = output[:expected_len]
            # ==============================================================

            output = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()

            assert(output.shape == label.shape)
            
            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]
                
                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        acc, edit, f1s = func_eval(
            label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    if use_east_proposal:
        proposal_dir_train = os.path.join(
            east_proposal_root,
            east_proposal_exp + "_train",
            f"split{split_id}",
            east_proposal_gpu_dir,
            "evaluation"
        )

        proposal_dir_test = os.path.join(
            east_proposal_root,
            east_proposal_exp + "_test",
            f"split{split_id}",
            east_proposal_gpu_dir,
            "evaluation"
        )

        print("EAST train proposal dir:", proposal_dir_train)
        print("EAST test proposal dir:", proposal_dir_test)

        if not os.path.isdir(proposal_dir_train):
            raise FileNotFoundError(f"Train proposal dir not found: {proposal_dir_train}")
        if not os.path.isdir(proposal_dir_test):
            raise FileNotFoundError(f"Test proposal dir not found: {proposal_dir_test}")
    else:
        proposal_dir_train = None
        proposal_dir_test = None
    feature_dir_768 = os.path.join(root_data_dir, dataset_name, 'breakfast_split3_breakfast_384_ep13')
    feature_dir_FLOW = os.path.join(root_data_dir, dataset_name, 'flow_features')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    train_data_dict = get_data_dict(
        feature_dir_RGB=feature_dir_768,
        feature_dir_FLOW=feature_dir_FLOW,
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        proposal_dir=proposal_dir_train
    )

    test_data_dict = get_data_dict(
        feature_dir_RGB=feature_dir_768,
        feature_dir_FLOW=feature_dir_FLOW,
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        proposal_dir=proposal_dir_test
    )
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
        log_freq=log_freq, log_train_results=log_train_results
    )