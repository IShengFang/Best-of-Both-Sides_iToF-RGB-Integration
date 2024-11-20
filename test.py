import argparse, yaml
import torch
import data, models, tester

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def load_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--data_root', type=str, default='ToFFlyingThings3D',
                        help='Directory path to ToFFlyingThings3D')
    parser.add_argument('--uw_weight_path', type=str)
    parser.add_argument('--w_weight_path', type=str)
    parser.add_argument('--stereo_weight_path', type=str)
    parser.add_argument('--config_path', type=str,
                        help='path to configuration yaml file')
    parser.add_argument('--test_cam', type=str, default='both',
                        choices=['both', 'uw', 'w', 'stereo'])
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='output_metric.txt',
                        help='output file name')

    parser.add_argument('--n_threads', type=int, default= 8)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = load_args()
    with open(args.config_path) as f:
        conf = yaml.safe_load(f)

    print('init models')
    if conf['mode'] == 'RGBD' or conf['mode'] == 'MonoRGB' or conf["mode"] == 'SplitStereoRGB':
        if args.test_cam=='both':
            test_models = dict()
            if conf["mode"]== 'RGBD':
                test_models['uw_model'] = models.model.RGBD_Model(**conf['model'])
                test_models['w_model'] = models.model.RGBD_Model(**conf['model'])
            elif conf["mode"] == 'MonoRGB':
                test_models['uw_model'] = models.model.MonoRGB_Model(**conf['model'])
                test_models['w_model'] = models.model.MonoRGB_Model(**conf['model'])
            elif conf["mode"] == 'SplitStereoRGB':
                test_models['uw_model'] = models.model.SplitStereoRGB_Model(**conf['model'])
                test_models['w_model'] = models.model.SplitStereoRGB_Model(**conf['model'])
            
            print('load uw model')
            test_models['uw_model'].load_state_dict(torch.load(args.uw_weight_path))
            test_models['uw_model'].to(device)
            test_models['uw_model'].eval()
            
            print('load w model')
            test_models['w_model'].load_state_dict(torch.load(args.w_weight_path))
            test_models['w_model'].to(device)
            test_models['w_model'].eval()

        elif args.test_cam == 'uw':
            if conf['mode'] == 'RGBD':
                test_model = models.model.RGBD_Model(**conf['model'])
            elif conf['mode'] == 'MonoRGB':
                test_model = models.model.MonoRGB_Model(**conf['model'])
            elif conf['mode'] == 'SplitStereoRGB':
                test_model = models.model.SplitStereoRGB_Model(**conf['model'])
            test_model.load_state_dict(torch.load(args.uw_weight_path))
            test_model.to(device)

        elif args.test_cam == 'w':
            if conf['mode'] == 'RGBD':
                test_model = models.model.RGBD_Model(**conf['model'])
            elif conf['mode'] == 'MonoRGB':
                test_model = models.model.MonoRGB_Model(**conf['model'])
            elif conf['mode'] == 'SplitStereoRGB':
                test_model = models.model.SplitStereoRGB_Model(**conf['model'])
            test_model.load_state_dict(torch.load(args.w_weight_path))
            test_model.to(device)

    elif conf['mode'] == 'StereoRGB':
            test_model = models.model.StereoRGB_Model(**conf['model'])
            test_model.load_state_dict(torch.load(args.stereo_weight_path))
            test_model.to(device)

    print('init tester')
    if conf['mode'] != 'StereoRGB':
        if args.test_cam == 'both':
            Tester = tester.BothModelTester(conf['data']['cam_path'], 
                                        conf['data']['depth_scale'], 
                                        test_models['uw_model'],
                                        test_models['w_model'],
                                        conf['mode']
                                        ).to(device)

        else:
            Tester = tester.SingleModelTester(conf['data']['cam_path'], 
                                              conf['data']['depth_scale'], 
                                              test_model,
                                              test_cam=args.test_cam,
                                              mode=conf['mode']
                                           ).to(device)
    elif conf['mode'] == 'StereoRGB':
        Tester = tester.StereoRGBTester(conf['data']['cam_path'], 
                                     conf['data']['depth_scale'],
                                     test_model
                                     ).to(device)
        
    print('init dataset')
    val_dataset  = data.ToF_FlyingTings3D_UW_W_Simple_Dataset(args.data_root, 'test')
    val_dataset  = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size=args.test_batch_size, 
                                               shuffle=False, 
                                               num_workers=args.n_threads, 
                                               drop_last=False)


    print('start testing')
    metrics = Tester.test(val_dataset)
    for key, value in metrics.items():
        print(key, value)
    with open(args.output, 'w') as f:
        for key, value in metrics.items():
            f.write(key + ': ' + str(value) + '\n')