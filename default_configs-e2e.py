import os
import json
import copy

params_gtea = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"gtea",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "use_east_proposal": True,
   "east_proposal_root": "/mnt/WRJ/EAST/EAST/exps/gtea/adatad",
   "east_proposal_exp": "e2e_actionformer_ret_ssv2_tpl_l16_25m_768x1_160_adapter3_2e-4_0.0002_p0.5",
   "east_proposal_gpu_dir": "gpu2_id0",
   "proposal_alpha": 0.5,
   "encoder_params":{
      "use_instance_norm":False, 
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":1152,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":  ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1'],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss_rgb":0.5,
      "encoder_mse_loss_rgb":0.025,
      "encoder_boundary_loss":0.0,
      "encoder_ce_loss_flow":0.5,
      "encoder_mse_loss_flow":0.025,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.025,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":1e-6,
   "num_epochs":10001,
   "log_freq":10,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":1,
   "soft_label": 1.4,
   "log_train_results":False,
   "postprocess":{
      "type":"purge",
      "value":3
   },
}

params_50salads = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"50salads",
   "split_id":1,
   "sample_rate":8,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":1152,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":7,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":1.0,
      "cond_types":[
         "full",
         "zero",
         "boundary05-",
         "segment=2",
         "segment=2"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss_rgb":0.5,
      "encoder_mse_loss_rgb":0.1,
      "encoder_boundary_loss":0.0,
      "encoder_ce_loss_flow":0.5,
      "encoder_mse_loss_flow":0.1, 
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":0,
   "num_epochs":5001,
   "log_freq":5,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":20,
   "soft_label": None,
   "log_train_results":False,
   "postprocess":{
      "type":"median", # W
      "value":30 # W
   },
}

params_breakfast = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"breakfast",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
#        以前是False
      "use_instance_norm":False,
      "num_layers":12,
      "num_f_maps":256,
      "input_dim":1152,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.1,
      "temporal_dropout_rate":0.1,
      "feature_layer_indices":[
         7,
         8,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":128,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "full",
         "zero",
         "boundary03-",
         "segment=1",
         "segment=1"
      ],
      "detach_decoder":False,
   },
   "loss_weights":{
      "encoder_ce_loss_rgb":0.1,
#        上面原来是0.5
      "encoder_mse_loss_rgb":0.025,
       #        原来是0.025
      "encoder_boundary_loss":0.0,
      "encoder_ce_loss_flow":0.9,
#        上面原来是0.5
      "encoder_mse_loss_flow":0.025,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.025,  
#        原来是0.025
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0001,
   "weight_decay":0,
   "num_epochs":1001,
   "log_freq":3,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":3,
   "soft_label":4,
   "log_train_results":False,
   "postprocess":{
      "type":"median",
      "value":15
   },
}

###################### GTEA #######################

split_num = 4

for split_id in range(1, split_num+1):
    
    params = copy.deepcopy(params_gtea)

    params['split_id'] = split_id
    params['naming'] = f'GTEA-siglipaim+i3d-eastprop-alpha05-S{split_id}'

    if not os.path.exists('configs'):
        os.makedirs('configs')
     
    file_name = os.path.join('configs', f'{params["naming"]}.json')

    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)


###################### 50salads #######################

split_num = 5

for split_id in range(1, split_num+1):
    
    params = copy.deepcopy(params_50salads)

    params['split_id'] = split_id
    params['naming'] = f'50salads-siglipaim+memi3d2048-S{split_id}'

    if not os.path.exists('configs'):
        os.makedirs('configs')
     
    file_name = os.path.join('configs', f'{params["naming"]}.json')

    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)

###################### Breakfast #######################

split_num = 4

for split_id in range(1, split_num+1):
    
    params = copy.deepcopy(params_breakfast)

    params['split_id'] = split_id
    params['naming'] = f'Breakfast-siglipaim+flow-rgbloss0.1-S{split_id}'

    if not os.path.exists('configs'):
        os.makedirs('configs')
     
    file_name = os.path.join('configs', f'{params["naming"]}.json')

    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)
