import argparse, os, yaml
import torch
from torch.utils.tensorboard import SummaryWriter

import data, models, trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def load_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--data_root', type=str, default='ToFFlyingThings3D/',
                        help='Directory path to ToFFlyingThings3D')

    parser.add_argument('--config_path', type=str, default='config/device_tft3d.yml',
                        help='path to configuration yaml file')

    parser.add_argument('--train_cam_model', type=str, default='both',
                        choices=['both', 'uw', 'w', 'stereo'])

    # training options
    parser.add_argument('--result_dir', default='results/',
                        help='Directory to save the results')

    parser.add_argument('--n_threads', type=int, default= 8)
                            
    #exp_name_option
    parser.add_argument('--generate_exp_name', action='store_true', default=False)

    
    args = parser.parse_args()

    return args

def generate_exp_name(args, conf):

    if conf['train']['use_mask']:
        conf['exp_name'] +='_w_mask' 

    conf['exp_name'] += f"_{conf['mode']}_{args.train_cam_model}"
        
    if conf['train']['two-stage']['loss']['structure_loss'] \
                    or conf['train']['loss']['structure_loss']:
        if conf['train']['structure_distillation']['knowledge']=='midas':
            conf['exp_name'] += f"_{conf['train']['structure_distillation']['midas']['midas_mode']}"
            if conf['train']['structure_distillation']['midas']['resize4midas']:
                conf['exp_name'] += '_r4m'
        

    if conf['train']['optim']['use_radam']:
        conf['exp_name']+='_radam'       


    conf['exp_name'] += '_res{}_lr_{}_bs_{}_{}_scale_{}'.format(
        conf['model']['resnet_depth'], 
        conf['train']['optim']['lr'],
        conf['train']['dataloader']['batch_size'],
        conf['data']['dataset'],
        conf['data']['depth_scale'])

    if conf['model']['pretrained_img_encoder']:
        conf['exp_name'] += '_PTEimg'

    if conf['model']['resblock_decoder']:
        conf['exp_name'] += '_ResD'


    if conf['train']['loss']['structure_loss']:
        conf['exp_name'] += f"_SD_{conf['train']['loss_weights']['structure_loss_weight']}"
    if conf['train']['loss']['depth_consist_loss']:
        conf['exp_name'] += f"_DC_{conf['train']['loss_weights']['depth_consist_weight']}"
    if conf['train']['loss']['smooth_loss']:
        conf['exp_name'] += f"_Sm_{conf['train']['loss_weights']['smooth_loss_weight']}"

    conf['exp_name'] += f"_E{conf['train']['epochs']}"

    if conf['train']['two-stage']['use']:
        conf['exp_name'] += f"_2st_E{conf['train']['two-stage']['epoch']}"

        if conf['train']['two-stage']['loss']['structure_loss']:
            conf['exp_name'] += f"_SD_{conf['train']['two-stage']['loss_weights']['structure_loss_weight']}"
        if conf['train']['two-stage']['loss']['depth_consist_loss']:
            conf['exp_name'] += f"_DC_{conf['train']['two-stage']['loss_weights']['depth_consist_weight']}"
        if conf['train']['two-stage']['loss']['smooth_loss']:
            conf['exp_name'] += f"_Sm_{conf['train']['two-stage']['loss_weights']['smooth_loss_weight']}"
    if conf['train']['optim']['use_lr_decay']:
        conf['exp_name'] += f"_lr_decay_{conf['train']['optim']['lr_decay']}"
    
    conf['exp_name'] = conf['exp_name'].replace('.', '_')

    return conf

def make_dirs(args, exp_name):
    log_input = {'tb_writer': os.path.join(args.result_dir, 'log', exp_name),
                   'img_dir': os.path.join(args.result_dir, 'img', exp_name),
                   'cpt_dir': os.path.join(args.result_dir, 'cpt', exp_name),
                    }

    for key, dir_path in log_input.items():
        os.makedirs(dir_path, exist_ok=True)
        
    return log_input

if __name__== "__main__":

    args = load_args()
    with open(args.config_path) as f:
        conf = yaml.safe_load(f)

    if args.generate_exp_name:
        conf = generate_exp_name(args, conf)

    print('start exp: ', conf['exp_name'])

    log_input = make_dirs(args, conf['exp_name'])

    if args.train_cam_model!='both':
        conf['train']['loss']['depth_consist_loss'] = False
        conf['train']['two-stage']['depth_consist_loss'] = False

    config_path = os.path.join(log_input['cpt_dir'], 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(conf, f)

    print('init tensorboad writer...')
    log_input['tb_writer'] = SummaryWriter(log_dir=log_input['tb_writer'])
    
    print('init model')
    if conf['mode'] == 'RGBD' or conf['mode'] == 'MonoRGB' or conf["mode"] == 'SplitStereoRGB':
        if args.train_cam_model=='both':
            train_models = dict()
            if conf["mode"]== 'RGBD':
                train_models['uw_model'] = models.model.RGBD_Model(**conf['model'])
                train_models['w_model'] = models.model.RGBD_Model(**conf['model'])
            elif conf["mode"] == 'MonoRGB':
                train_models['uw_model'] = models.model.MonoRGB_Model(**conf['model'])
                train_models['w_model'] = models.model.MonoRGB_Model(**conf['model'])
            elif conf["mode"] == 'SplitStereoRGB':
                train_models['uw_model'] = models.model.SplitStereoRGB_Model(**conf['model'])
                train_models['w_model'] = models.model.SplitStereoRGB_Model(**conf['model'])
        else:
            if conf["mode"]== 'RGBD':
                model = models.model.RGBD_Model(**conf['model'])
            elif conf["mode"] == 'MonoRGB':
                model = models.model.MonoRGB_Model(**conf['model'])
            elif conf["mode"] == 'SplitStereoRGB':
                model = models.model.SplitStereoRGB_Model(**conf['model'])

    elif conf['mode'] == 'StereoRGB':
            model = models.model.StereoRGB_Model(**conf['model'])


    if conf['data']['dataset'] == 'TFT3D':
        print('init TFT3D dataloader...')
        train_dataset = data.ToF_FlyingTings3D_UW_W_Simple_Dataset(args.data_root, 'train')
        val_dataset  = data.ToF_FlyingTings3D_UW_W_Simple_Dataset(args.data_root, 'val')
        train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=conf['train']['dataloader']['batch_size'], shuffle=True, num_workers=args.n_threads)
        val_dataset  = torch.utils.data.DataLoader(val_dataset, batch_size=conf['train']['dataloader']['val_batch_size'], shuffle=False, num_workers=args.n_threads, drop_last=True) 
    else:
        raise NotImplementedError
     
    datasets = {'train_dataset':train_dataset, 'val_dataset':val_dataset}
    
    print('init Trainer')
    if conf['data']['dataset'] == 'TFT3D':
        if args.train_cam_model=='both':
            if conf["mode"]=='SplitStereoRGB':
                Trainer = trainer.TFT3D_TwoSplitStereoModelTrainer(**train_models)
                Trainer.init_other_settings(conf['train']['use_mask'], conf['data']['depth_scale'], conf['mode'], 
                                            conf['train']['structure_distillation']['knowledge'])
            elif conf["mode"]=='StereoRGB':
                Trainer = trainer.TFT3D_StereoModelTrainer(model)
                Trainer.init_other_settings(conf['train']['use_mask'], conf['data']['depth_scale'], conf['mode'], 
                                            conf['train']['structure_distillation']['knowledge'])
            else:
                Trainer = trainer.TFT3D_TwoSplitModelTrainer(**train_models)
                Trainer.init_other_settings(conf['train']['use_mask'], conf['data']['depth_scale'], conf['mode'], 
                                            conf['train']['structure_distillation']['knowledge'])
        elif args.train_cam_model=='uw' or args.train_cam_model=='w':
            if conf["mode"]=='SplitStereoRGB':
                Trainer = trainer.TFT3D_SingleSplitStereoModelTrainer(model, args.train_cam_model)
                Trainer.init_other_settings(conf['train']['use_mask'], conf['data']['depth_scale'], conf['mode'],
                                            conf['train']['structure_distillation']['knowledge'])
            else:
                Trainer = trainer.TFT3D_SingleSplitModelTrainer(model, args.train_cam_model)
                Trainer.init_other_settings(conf['train']['use_mask'], conf['data']['depth_scale'], conf['mode'],
                                            conf['train']['structure_distillation']['knowledge'])
        elif args.train_cam_model == 'stereo':
            if conf["mode"]=='StereoRGB':
                Trainer = trainer.TFT3D_StereoModelTrainer(model)
                Trainer.init_other_settings(conf['train']['use_mask'], conf['data']['depth_scale'], conf['mode'],
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        
    print('  init log paths')
    Trainer.init_log(**log_input)
    print('  load camera matrix')
    Trainer.load_camera_matrix(conf['data']['cam_path'])
    print('  load dataloader')
    Trainer.load_dataset(**datasets)
    print('  init loss setting...')
    Trainer.init_loss(**conf['train']['loss'])
    print('  init loss weights...')
    Trainer.init_loss_weights(**conf['train']['loss_weights'])
    if conf['train']['two-stage']['use']:
        print('  init two stage training')
        Trainer.init_two_stage_setting(conf['train']['two-stage'])
    if conf['train']['loss']['structure_loss'] \
        or conf['train']['two-stage']['loss']['structure_loss']:
        print('  init structure knowledge')
        if conf['train']['structure_distillation']['knowledge'] == 'midas':
            print(f'    load {conf["train"]["structure_distillation"]["midas"]["midas_mode"]} model...')
            Trainer.init_midas(**conf["train"]["structure_distillation"]["midas"])

    print('  init test data for visualization')
    Trainer.init_val_data()
    print('  init optimizer')
    Trainer.init_optim(**conf['train']['optim'])
    print('strat fitting...')
    Trainer.fit(conf['train']['epochs'])