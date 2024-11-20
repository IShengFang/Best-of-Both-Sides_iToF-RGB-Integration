import os, pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from tqdm import tqdm, trange
import kornia as K

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

EPS = 1e-8

class BasicTrainer(nn.Module):
    ssim_l1_alpha = 0.85   
    train_w_two_stage = False
    in_second_stage = False
    uw_best_mae_epoch = False 
    w_best_mae_epoch = False 
    uw_best_mae = 9487
    w_best_mae = 9487

    def _smooth_loss(self, depth, image, beta=1.0):
        """
        Calculate the image-edge-aware second-order smoothness loss for flo 
        modified from https://github.com/lelimite4444/BridgeDepthFlow/blob/14183b99830e1f41e774c0d43fdb058d07f2a397/utils/utils.py#L60
        """
        
        img_grad_x, img_grad_y = self._gradient(image)
        weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
        weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

        dx, dy = self._gradient(depth)

        dx2, dxdy = self._gradient(dx)
        dydx, dy2 = self._gradient(dy)
        del dxdy, dydx

        return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

    def _gradient(self, pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def _ssim_loss(self, x, y):
        '''
        from monodepth 
        https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/monodepth_model.py#L91
        '''
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)
        
        #(input, kernel, stride, padding)
        sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
        
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        SSIM = SSIM_n / SSIM_d
        
        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def load_camera_matrix(self, cam_path):

        np.seterr(divide='ignore',invalid='ignore')
        
        with open(os.path.join(cam_path, 'W_calib.pkl'), 'rb') as f:
            self.W_calib = pickle.load(f)
        with open(os.path.join(cam_path, 'UW_calib.pkl'), 'rb') as f:
            self.UW_calib = pickle.load(f)
        with open(os.path.join(cam_path, 'I_calib.pkl'), 'rb') as f:
            self.I_calib = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_I_UW.pkl'),'rb') as f:
            self.I_UW_pose = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_I_W.pkl'),'rb') as f:
            self.I_W_pose = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_UW_I.pkl'),'rb') as f:
            self.UW_I_pose = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_UW_W.pkl'),'rb') as f:
            self.UW_W_pose = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_W_I.pkl'),'rb') as f:
            self.W_I_pose = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_W_UW.pkl'),'rb') as f:
            self.W_UW_pose = pickle.load(f)

        self.focal_length_x = self.UW_calib['mtx'][0,0]
        self.baseline = 10.0

        S_I_UW = self.I_calib['mtx']/self.UW_calib['mtx']
        self.scale_I_UW = (S_I_UW[0,0]+ S_I_UW[1,1])/2
        S_I_W = self.I_calib['mtx']/self.W_calib['mtx']
        self.scale_I_W = (S_I_W[0,0]+ S_I_W[1,1])/2
        S_W_UW = self.W_calib['mtx']/self.UW_calib['mtx']
        self.scale_W_UW = (S_W_UW[0,0]+ S_W_UW[1,1])/2

        Rt_I_UW = np.hstack((self.I_UW_pose['R'], self.I_UW_pose['t']))
        Rt_I_W = np.hstack((self.I_W_pose['R'], self.I_W_pose['t']))
        Rt_UW_I = np.hstack((self.UW_I_pose['R'], self.UW_I_pose['t']))
        Rt_W_I = np.hstack((self.W_I_pose['R'], self.W_I_pose['t']))
        Rt_W_UW = np.hstack((self.W_UW_pose['R'], self.W_UW_pose['t']))
        Rt_UW_W = np.hstack((self.UW_W_pose['R'], self.UW_W_pose['t']))
        self.Rt_I_UW = torch.tensor(np.vstack((Rt_I_UW, np.array([[0,0,0,1]])))).unsqueeze(0).float()
        self.Rt_I_W = torch.tensor(np.vstack((Rt_I_W, np.array([[0,0,0,1]])))).unsqueeze(0).float()
        self.Rt_UW_I = torch.tensor(np.vstack((Rt_UW_I, np.array([[0,0,0,1]])))).unsqueeze(0).float()
        self.Rt_W_I = torch.tensor(np.vstack((Rt_W_I, np.array([[0,0,0,1]])))).unsqueeze(0).float()
        self.Rt_W_UW = torch.tensor(np.vstack((Rt_W_UW, np.array([[0,0,0,1]])))).unsqueeze(0).float()
        self.Rt_UW_W = torch.tensor(np.vstack((Rt_UW_W, np.array([[0,0,0,1]])))).unsqueeze(0).float()

        self.K_I  = torch.tensor(self.I_calib['mtx']).unsqueeze(0).float()
        self.K_W  = torch.tensor(self.W_calib['mtx']).unsqueeze(0).float()
        self.K_UW = torch.tensor(self.UW_calib['mtx']).unsqueeze(0).float()

    def UWwarp2W_by_UW_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        UW_3d_pts = K.geometry.depth_to_3d(depth, self.K_UW.repeat(B, 1, 1).cuda(), normalize_points=True)
        UW_3d_pts = UW_3d_pts.permute(0, 2, 3, 1)
        UW_3d_pts_W = K.geometry.transform_points(self.Rt_UW_W.repeat(B, 1, 1, 1).cuda(), UW_3d_pts)
        UW_2d_pts_W = K.geometry.project_points(UW_3d_pts_W.reshape(B, H * W, 3), 
                                                self.K_W.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        W2UW_warp_grid = K.geometry.normalize_pixel_coordinates(UW_2d_pts_W, H, W)
        W2UW_flow = W2UW_warp_grid - mesh_grid
        UW2W_warp_grid = mesh_grid - W2UW_flow / self.scale_W_UW
        warped = F.grid_sample(warp_src, UW2W_warp_grid, align_corners=True)
        return warped

    def Wwarp2UW_by_W_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        W_3d_pts = K.geometry.depth_to_3d(depth, self.K_W.repeat(B, 1, 1).cuda(), normalize_points=True)
        W_3d_pts = W_3d_pts.permute(0, 2, 3, 1)
        W_3d_pts_UW = K.geometry.transform_points(self.Rt_W_UW.repeat(B, 1, 1, 1).cuda(), W_3d_pts)
        W_2d_pts_UW = K.geometry.project_points(W_3d_pts_UW.reshape(B, H * W, 3), 
                                                self.K_UW.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        UW2W_warp_grid = K.geometry.normalize_pixel_coordinates(W_2d_pts_UW, H, W)
        UW2W_flow = UW2W_warp_grid - mesh_grid
        W2UW_warp_grid = mesh_grid - UW2W_flow * self.scale_W_UW
        warped = F.grid_sample(warp_src, W2UW_warp_grid, align_corners=True)
        return warped

    def Wwarp2UW_by_UW_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        UW_3d_pts = K.geometry.depth_to_3d(depth, self.K_UW.repeat(B, 1, 1).cuda(), normalize_points=True)
        UW_3d_pts = UW_3d_pts.permute(0, 2, 3, 1)
        UW_3d_pts_W = K.geometry.transform_points(self.Rt_UW_W.repeat(B, 1, 1, 1).cuda(), UW_3d_pts)
        UW_2d_pts_W = K.geometry.project_points(UW_3d_pts_W.reshape(B, H * W, 3), 
                                                self.K_W.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        W2UW_warp_grid = K.geometry.normalize_pixel_coordinates(UW_2d_pts_W, H, W)
        warped = F.grid_sample(warp_src, W2UW_warp_grid, align_corners=True)
        return warped

    def UWwarp2W_by_W_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        W_3d_pts = K.geometry.depth_to_3d(depth, self.K_W.repeat(B, 1, 1).cuda(), normalize_points=True)
        W_3d_pts = W_3d_pts.permute(0, 2, 3, 1)
        W_3d_pts_UW = K.geometry.transform_points(self.Rt_W_UW.repeat(B, 1, 1, 1).cuda(), W_3d_pts)
        W_2d_pts_UW = K.geometry.project_points(W_3d_pts_UW.reshape(B, H * W, 3), 
                                                self.K_UW.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        UW2W_warp_grid = K.geometry.normalize_pixel_coordinates(W_2d_pts_UW, H, W)
        warped = F.grid_sample(warp_src, UW2W_warp_grid, align_corners=True)
        return warped

    def UWwarp2I_by_UW_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        UW_3d_pts = K.geometry.depth_to_3d(depth, self.K_UW.repeat(B, 1, 1).cuda(), normalize_points=True)
        UW_3d_pts = UW_3d_pts.permute(0, 2, 3, 1)
        UW_3d_pts_I = K.geometry.transform_points(self.Rt_UW_I.repeat(B, 1, 1, 1).cuda(), UW_3d_pts)
        UW_2d_pts_I = K.geometry.project_points(UW_3d_pts_I.reshape(B, H * W, 3), 
                                                self.K_I.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        I2UW_warp_grid = K.geometry.normalize_pixel_coordinates(UW_2d_pts_I, H, W)
        I2UW_flow = I2UW_warp_grid - mesh_grid
        UW2I_warp_grid = mesh_grid - I2UW_flow / self.scale_I_UW
        warped = F.grid_sample(warp_src, UW2I_warp_grid, align_corners=True)
        return warped

    def Iwarp2UW_by_I_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        I_3d_pts = K.geometry.depth_to_3d(depth, self.K_I.repeat(B, 1, 1).cuda(), normalize_points=True)
        I_3d_pts = I_3d_pts.permute(0, 2, 3, 1)
        I_3d_pts_UW = K.geometry.transform_points(self.Rt_I_UW.repeat(B, 1, 1, 1).cuda(), I_3d_pts)
        I_2d_pts_UW = K.geometry.project_points(I_3d_pts_UW.reshape(B, H * W, 3),
                                                self.K_UW.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        UW2I_warp_grid = K.geometry.normalize_pixel_coordinates(I_2d_pts_UW, H, W)
        UW2I_flow = UW2I_warp_grid-mesh_grid
        I2UW_warp_grid = mesh_grid-UW2I_flow*self.scale_I_UW
        warped = F.grid_sample(warp_src, I2UW_warp_grid, align_corners=True)
        return warped

    def Iwarp2W_by_I_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        I_3d_pts = K.geometry.depth_to_3d(depth, self.K_I.repeat(B, 1, 1).cuda(), normalize_points=True)
        I_3d_pts = I_3d_pts.permute(0, 2, 3, 1)
        I_3d_pts_W = K.geometry.transform_points(self.Rt_I_W.repeat(B, 1, 1, 1).cuda(), I_3d_pts)
        I_2d_pts_W = K.geometry.project_points(I_3d_pts_W.reshape(B, H * W, 3), 
                                               self.K_W.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        W2I_warp_grid = K.geometry.normalize_pixel_coordinates(I_2d_pts_W, H, W)
        W2I_flow = W2I_warp_grid-mesh_grid
        I2W_warp_grid = mesh_grid-W2I_flow*self.scale_I_W
        warped = F.grid_sample(warp_src, I2W_warp_grid, align_corners=True)
        return warped

    def _normalize_depth(self, depth):
        return depth/(depth.mean(3, True).mean(2, True)+ EPS)

    def _ssim_l1_loss(self, target, predict):
        return self.ssim_l1_alpha*self._ssim_loss(target,predict).mean()+\
               (1-self.ssim_l1_alpha)*F.l1_loss(target,predict)

    def init_midas(self, midas_mode, resize4midas):
        if self.structure_loss:
            self.midas_mode = midas_mode
            self.midas = torch.hub.load("intel-isl/MiDaS", self.midas_mode).cuda()
            self.resize4midas = resize4midas
            self.midas_transform = []
            if self.resize4midas:
                if self.midas_mode[-5:] =='small':
                    self.midas_transform += [torchvision.transforms.Resize(256, torchvision.transforms.InterpolationMode.BICUBIC),]
                else:
                    self.midas_transform += [torchvision.transforms.Resize(384, torchvision.transforms.InterpolationMode.BICUBIC),]
            
            if self.midas_mode[:3] =='DPT':
                self.midas_transform += [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            else:
                self.midas_transform += [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

            self.midas_transform = torchvision.transforms.Compose(self.midas_transform)

    def load_dataset(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def init_loss(self,photo_loss, structure_loss, smooth_loss, depth_consist_loss):
        self.photo_loss = photo_loss
        self.structure_loss = structure_loss
        self.smooth_loss = smooth_loss
        self.depth_consist_loss = depth_consist_loss

    def init_loss_weights(self, warp2uw_weight, warp2w_weight, 
                                smooth_loss_weight, structure_loss_weight, depth_consist_weight):

        self.weights = dict()

        for index in range(5):
            if self.photo_loss =='l1':
                self.weights[f'UW_l1_warp_loss_by_UW_depth_{index}'] = warp2uw_weight
                self.weights[f'UW_l1_warp_loss_by_W_depth_{index}'] = warp2uw_weight
                self.weights[f'W_l1_warp_loss_by_UW_depth_{index}'] = warp2w_weight
                self.weights[f'W_l1_warp_loss_by_W_depth_{index}'] = warp2w_weight
            elif self.photo_loss =='l2':
                self.weights[f'UW_l2_warp_loss_by_UW_depth_{index}'] = warp2uw_weight
                self.weights[f'UW_l2_warp_loss_by_W_depth_{index}'] = warp2uw_weight
                self.weights[f'W_l2_warp_loss_by_UW_depth_{index}'] = warp2w_weight
                self.weights[f'W_l2_warp_loss_by_W_depth_{index}'] = warp2w_weight
            elif self.photo_loss == 'ssim':
                self.weights[f'UW_ssim_warp_loss_by_UW_depth_{index}'] = warp2uw_weight
                self.weights[f'UW_ssim_warp_loss_by_W_depth_{index}'] = warp2uw_weight
                self.weights[f'W_ssim_warp_loss_by_UW_depth_{index}'] = warp2w_weight
                self.weights[f'W_ssim_warp_loss_by_W_depth_{index}'] = warp2w_weight
            elif self.photo_loss == 'ssim_l1':
                self.weights[f'UW_ssim_l1_warp_loss_by_UW_depth_{index}'] = warp2uw_weight
                self.weights[f'UW_ssim_l1_warp_loss_by_W_depth_{index}'] = warp2uw_weight
                self.weights[f'W_ssim_l1_warp_loss_by_UW_depth_{index}'] = warp2w_weight
                self.weights[f'W_ssim_l1_warp_loss_by_W_depth_{index}'] = warp2w_weight

            if self.structure_loss:
                self.weights[f'UW_structure_ssim_loss_{index}'] = structure_loss_weight
                self.weights[f'W_structure_ssim_loss_{index}']  = structure_loss_weight

            if self.smooth_loss:
                self.weights[f'UW_smooth_loss_{index}'] = smooth_loss_weight
                self.weights[f'W_smooth_loss_{index}'] = smooth_loss_weight

            if self.depth_consist_loss =='l1':
                self.weights[f'UW_l1_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'UW_l1_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
                self.weights[f'W_l1_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'W_l1_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
            elif self.depth_consist_loss =='l2':
                self.weights[f'UW_l2_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'UW_l2_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
                self.weights[f'W_l2_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'W_l2_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
            elif self.depth_consist_loss == 'ssim':
                self.weights[f'UW_ssim_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'UW_ssim_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
                self.weights[f'W_ssim_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'W_ssim_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
            elif self.depth_consist_loss == 'ssim_l1':
                self.weights[f'UW_ssim_l1_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'UW_ssim_l1_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight
                self.weights[f'W_ssim_l1_depth_consist_loss_by_UW_depth_{index}'] = depth_consist_weight
                self.weights[f'W_ssim_l1_depth_consist_loss_by_W_depth_{index}'] = depth_consist_weight


    def compute_depth_metrics(self, predict, ground_truth):
        '''
        borrow by https://github.com/dusty-nv/pytorch-depth/blob/master/metrics.py
        '''
        valid_mask = ground_truth>0
        predict = predict[valid_mask]
        ground_truth = ground_truth[valid_mask]

        abs_diff = (predict - ground_truth).abs()
        mse = torch.pow(abs_diff, 2).mean()
        rmse = torch.sqrt(mse)
        mae = abs_diff.mean()
        log_diff = torch.log10(predict) - torch.log10(ground_truth)
        lg10 = log_diff.abs().mean()
        rmse_log = torch.sqrt(torch.pow(log_diff, 2).mean())
        absrel = float((abs_diff / ground_truth).mean())
        sqrel = float((torch.pow(abs_diff, 2) / ground_truth).mean())

        maxRatio = torch.max(predict / ground_truth, ground_truth / predict)
        delta1 = (maxRatio < 1.25).float().mean()
        delta2 = (maxRatio < 1.25 ** 2).float().mean()
        delta3 = (maxRatio < 1.25 ** 3).float().mean()
        return mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log



    def _adjust_learning_rate(self, ):
        lr = self.lr / (1.0 + self.lr_decay * self.n_iter)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def init_two_stage_setting(self, setting):
        self.train_w_two_stage = setting['use']
        self.first_stage_epoch = setting['epoch']
        self.second_stage_loss_setting = setting['loss']
        self.second_stage_weight_setting = setting['loss_weights']

    def update_two_stage_loss_setting(self):
        if self.epoch >= self.first_stage_epoch:
            print('update second stage loss setting...')
            self.init_loss(**self.second_stage_loss_setting)
            self.init_loss_weights(**self.second_stage_weight_setting)
            self.in_second_stage=True

    def fit(self, epochs):
        for epoch in tqdm(range(epochs)):
            if self.train_w_two_stage and not self.in_second_stage:
                self.update_two_stage_loss_setting()
            self._epoch()
        self._save_model_cpt()

    def _visualize_depths(self, depth_tag, depth, save_path):
        B, C, H, W = depth.size()
        depth = depth.detach() * self.depth_scale
        depth_tag = depth_tag.split('/')
        for i in range(B):
            depth_i = depth[i].squeeze(0).cpu().numpy()
            depth_figure = plt.figure(figsize=(6, 4))
            plt.imshow(depth_i, cmap='plasma_r', vmin=0, vmax=self.depth_scale)
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f'{depth_tag[-1]}_{i}.png'), dpi=100)
            self.tb_writer.add_figure(f'{depth_tag[0]}/{depth_tag[-1]}_{i}', depth_figure, self.n_iter)
            plt.clf() 

    def _visualize_structure_depths(self, depth_tag, depth, save_path):
        B, C, H, W = depth.size()
        depth = depth.detach()
        depth_tag = depth_tag.split('/')
        for i in range(B):
            depth_i = depth[i].squeeze(0).cpu().numpy()
            depth_figure = plt.figure(figsize=(6, 4))
            plt.imshow(depth_i, cmap='plasma_r', vmin=0, vmax=1.)
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f'{depth_tag[-1]}_{i}.png'), dpi=100)
            self.tb_writer.add_figure(f'{depth_tag[0]}/{depth_tag[-1]}_{i}', depth_figure, self.n_iter)
            plt.clf() 

    def init_log(self, tb_writer, img_dir, cpt_dir):
        self.tb_writer = tb_writer
        self.img_dir = img_dir
        self.cpt_dir = cpt_dir
    
    def init_other_settings(self,use_mask, depth_scale, mode, struct_knowledge_src):
        self.mode = mode
        self.use_mask = use_mask
        self.depth_scale = depth_scale
        self.struct_knowledge_src = struct_knowledge_src
    def __init__(self, ):
        super().__init__()

    def init_optim(self,):
        raise NotImplementedError

    def init_val_data(self,):
        raise NotImplementedError

    def _compute_losses(self,):
        raise NotImplementedError

    def _compute_val_metrics(self,):
        raise NotImplementedError

    def _epoch_fit(self, ):
        raise NotImplementedError

    def _epoch_val(self,):
        raise NotImplementedError

    def _epoch(self,):
        raise NotImplementedError

    def _visualize(self,):
        raise NotImplementedError

    def _save_model_cpt(self,):
        raise NotImplementedError

class TFT3D_TwoSplitModelTrainer(BasicTrainer):
    def __init__(self, uw_model, w_model):
        super().__init__()
        self.uw_model  = uw_model.cuda() 
        self.w_model  = w_model.cuda() 

        self.n_iter = 0
        self.epoch = 0

    def init_optim(self, lr, betas, lr_decay, use_radam, use_lr_decay,
                   fixed_img_encoder ):
        print('init optimizer....')
        self.lr = lr 
        self.betas = betas
        self.use_radam = use_radam
        self.lr_decay = lr_decay
        self.use_lr_decay = use_lr_decay

        self.fixed_img_encoder = fixed_img_encoder

        update_models = []
        if fixed_img_encoder:
            update_models.append(self.uw_model.D)
            update_models.append(self.w_model.D)
        else:
            update_models.append(self.uw_model)
            update_models.append(self.w_model)

        update_models = nn.ModuleList(update_models)

        if self.use_radam:
            self.opt = torch.optim.RAdam(update_models.parameters(), lr=self.lr, betas=self.betas)
        else:
            self.opt = torch.optim.Adam(update_models.parameters(), lr=self.lr, betas=self.betas)

    def init_val_data(self,):
        gt_depth_uw, rgb_uw, ndepth_uw, conf_uw = next(iter(self.val_dataset))
        gt_depth_uw = gt_depth_uw.cuda()
        rgb_uw = rgb_uw.cuda()
        conf_uw = conf_uw.cuda()
        self.val_conf_uw = conf_uw.cpu()

        if 'RGBD' in self.mode:
            ndepth_uw = ndepth_uw.cuda()
        else:
            ndepth_uw = None

        tof_depth = None
        input_depth_uw = None
        input_depth_w = None
        self.val_input_depth_uw = None
        self.val_input_depth_w = None


        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)
        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)
        self.val_conf_w = conf_w.cpu()

        if tof_depth is not None:
            input_depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            input_depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            self.val_input_depth_uw = input_depth_uw.cpu()
            self.val_input_depth_w = input_depth_w.cpu()


        self.val_rgb_uw = rgb_uw.cpu()
        self.val_rgb_w = rgb_w.cpu()

        gt_depth_tof = self.UWwarp2I_by_UW_depth(gt_depth_uw, gt_depth_uw)

        save_path = os.path.join(self.img_dir, 'ground_truth')
        os.makedirs(save_path, exist_ok=True)

        if tof_depth is not None:
            tof_depth /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/tof_depth', tof_depth, self.n_iter)
            torchvision.utils.save_image(tof_depth, os.path.join(save_path, f'tof_depth.png')) 
            self._visualize_depths('ground_truth/tof_depth', tof_depth, save_path)

            self.tb_writer.add_images(f'input/depth_uw', self.val_input_depth_uw, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_uw, os.path.join(save_path, f'input_depth_uw.png')) 
            self._visualize_depths('input/depth_uw', self.val_input_depth_uw, save_path)

            self.tb_writer.add_images(f'input/depth_w', self.val_input_depth_w, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_w, os.path.join(save_path, f'input_depth_w.png'))
            self._visualize_depths('input/depth_w', self.val_input_depth_w, save_path)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw) * self.depth_scale
            mask_uw = self.Wwarp2UW_by_UW_depth(mask_uw, mask_uw)>0
            mask_uw = mask_uw.float()
            self.tb_writer.add_images(f'input/mask_uw', mask_uw, self.n_iter)
            torchvision.utils.save_image(mask_uw, os.path.join(save_path, f'mask_uw.png'))

        self.tb_writer.add_images(f'ground_truth/rgb_uw', rgb_uw, self.n_iter)
        torchvision.utils.save_image(rgb_uw, os.path.join(save_path, f'rgb_uw.png')) 
        self.tb_writer.add_images(f'ground_truth/rgb_w', rgb_w, self.n_iter)
        torchvision.utils.save_image(rgb_w, os.path.join(save_path, f'rgb_w.png'))     

        gt_depth_uw /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/gt_depth_uw', gt_depth_uw, self.n_iter)
        torchvision.utils.save_image(gt_depth_uw, os.path.join(save_path, f'gt_depth_uw.png')) 
        self._visualize_depths('gt_depth_uw/depth_uw', gt_depth_uw, save_path)

        gt_depth_uw_conf = gt_depth_uw * conf_uw
        self.tb_writer.add_images(f'ground_truth/gt_depth_uw_conf', gt_depth_uw_conf, self.n_iter)
        torchvision.utils.save_image(gt_depth_uw_conf, os.path.join(save_path, f'gt_depth_uw_conf.png')) 
        self._visualize_depths('gt_depth_uw/depth_uw_conf', gt_depth_uw_conf, save_path)

        gt_depth_tof /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/gt_depth_tof', gt_depth_tof, self.n_iter)
        torchvision.utils.save_image(gt_depth_tof, os.path.join(save_path, f'gt_depth_tof.png')) 
        self._visualize_depths('gt_depth_tof/depth_tof', gt_depth_tof, save_path)

        gt_depth_w /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/gt_depth_w', gt_depth_w, self.n_iter)
        torchvision.utils.save_image(gt_depth_w, os.path.join(save_path, f'gt_depth_w.png')) 
        self._visualize_depths('gt_depth_w/depth_w', gt_depth_w, save_path)

        gt_depth_w_conf = gt_depth_w * conf_w
        self.tb_writer.add_images(f'ground_truth/gt_depth_w_conf', gt_depth_w_conf, self.n_iter)
        torchvision.utils.save_image(gt_depth_w_conf, os.path.join(save_path, f'gt_depth_w.png')) 
        self._visualize_depths('gt_depth_w/depth_w_conf', gt_depth_w, save_path)

        if self.structure_loss:
            with torch.no_grad():
                if self.struct_knowledge_src == 'midas':
                    monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                    monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                    monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    monodepth_uw = 1 - monodepth_uw
                    monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    monodepth_w = 1 - monodepth_w

                    if self.resize4midas:
                        monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                        mode="bicubic")
                        monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                        mode="bicubic")

                    self.tb_writer.add_images(f'structure_depth/depth_uw', monodepth_uw, self.n_iter)
                    torchvision.utils.save_image(monodepth_uw, os.path.join(save_path, f'midas_uw.png')) 
                    self._visualize_structure_depths('structure_depth/depth_uw', monodepth_uw, save_path)

                    self.tb_writer.add_images(f'structure_depth/depth_w', monodepth_w, self.n_iter)
                    torchvision.utils.save_image(monodepth_w, os.path.join(save_path, f'midas_w.png')) 
                    self._visualize_structure_depths('structure_depth/depth_w', monodepth_w, save_path)


    def _compute_losses(self, depths_pred_uw, depths_pred_w, \
                        rgb_uw, rgb_w, monodepth_uw, monodepth_w, \
                        mask_uw):
        losses = dict()

        for index in range(5):
            depth_i_pred_uw = depths_pred_uw[index]
            depth_i_pred_w = depths_pred_w[index]

            rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, rgb_w)
            rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, rgb_uw)

            rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, rgb_w)
            rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, rgb_uw)

            if self.photo_loss =='l1':
                losses[f'UW_l1_warp_loss_by_UW_depth_{index}'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw)
                losses[f'UW_l1_warp_loss_by_W_depth_{index}'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw)
                losses[f'W_l1_warp_loss_by_UW_depth_{index}'] = F.l1_loss(rgb_w,rgb_w_pred_by_depth_i_uw)
                losses[f'W_l1_warp_loss_by_W_depth_{index}'] = F.l1_loss(rgb_w,rgb_w_pred_by_depth_i_w)
            elif self.photo_loss =='l2':
                losses[f'UW_l2_warp_loss_by_UW_depth_{index}'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw)
                losses[f'UW_l2_warp_loss_by_W_depth_{index}'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw)
                losses[f'W_l2_warp_loss_by_UW_depth_{index}'] = F.mse_loss(rgb_w,rgb_w_pred_by_depth_i_uw)
                losses[f'W_l2_warp_loss_by_W_depth_{index}'] = F.mse_loss(rgb_w,rgb_w_pred_by_depth_i_w)
            elif self.photo_loss == 'ssim':
                losses[f'UW_ssim_warp_loss_by_UW_depth_{index}'] = self._ssim_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw).mean()
                losses[f'UW_ssim_warp_loss_by_W_depth_{index}'] = self._ssim_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw).mean()
                losses[f'W_ssim_warp_loss_by_UW_depth_{index}'] = self._ssim_loss(rgb_w,rgb_w_pred_by_depth_i_uw).mean()
                losses[f'W_ssim_warp_loss_by_W_depth_{index}'] = self._ssim_loss(rgb_w,rgb_w_pred_by_depth_i_w).mean()
            elif self.photo_loss == 'ssim_l1':
                losses[f'UW_ssim_l1_warp_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw).mean()
                losses[f'UW_ssim_l1_warp_loss_by_W_depth_{index}'] = self._ssim_l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw).mean()
                losses[f'W_ssim_l1_warp_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(rgb_w,rgb_w_pred_by_depth_i_uw).mean()
                losses[f'W_ssim_l1_warp_loss_by_W_depth_{index}'] = self._ssim_l1_loss(rgb_w,rgb_w_pred_by_depth_i_w).mean()

            if self.smooth_loss:
                losses[f'UW_smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_uw, rgb_uw)
                losses[f'W_smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_w, rgb_w)

            if self.structure_loss:
                losses[f'UW_structure_ssim_loss_{index}'] = self._ssim_loss(self._normalize_depth(depth_i_pred_uw), self._normalize_depth(monodepth_uw)).mean()
                losses[f'W_structure_ssim_loss_{index}']  = self._ssim_loss(self._normalize_depth(depth_i_pred_w), self._normalize_depth(monodepth_w)).mean()

            if self.depth_consist_loss:
                detached_depth_i_pred_uw = depth_i_pred_uw.detach()
                depth_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(detached_depth_i_pred_uw*self.depth_scale, depth_i_pred_w)
                depth_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, depth_i_pred_w)
                depth_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, detached_depth_i_pred_uw)
                depth_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(detached_depth_i_pred_uw*self.depth_scale, detached_depth_i_pred_uw)


                if self.depth_consist_loss =='l1':
                    losses[f'UW_l1_depth_consist_loss_by_UW_depth_{index}'] = F.l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_l1_depth_consist_loss_by_W_depth_{index}']  = F.l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_l1_depth_consist_loss_by_UW_depth_{index}']  = F.l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_l1_depth_consist_loss_by_W_depth_{index}']   = F.l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)
                elif self.depth_consist_loss =='l2':
                    losses[f'UW_l2_depth_consist_loss_by_UW_depth_{index}'] = F.mse_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_l2_depth_consist_loss_by_W_depth_{index}']  = F.mse_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_l2_depth_consist_loss_by_UW_depth_{index}']  = F.mse_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_l2_depth_consist_loss_by_W_depth_{index}']   = F.mse_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)
                elif self.depth_consist_loss == 'ssim':
                    losses[f'UW_ssim_depth_consist_loss_by_UW_depth_{index}'] = self._ssim_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_ssim_depth_consist_loss_by_W_depth_{index}']  = self._ssim_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_ssim_depth_consist_loss_by_UW_depth_{index}']  = self._ssim_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_ssim_depth_consist_loss_by_W_depth_{index}'  ] = self._ssim_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)
                elif self.depth_consist_loss == 'ssim_l1':
                    losses[f'UW_ssim_l1_depth_consist_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_ssim_l1_depth_consist_loss_by_W_depth_{index}']  = self._ssim_l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_ssim_l1_depth_consist_loss_by_UW_depth_{index}']  = self._ssim_l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_ssim_l1_depth_consist_loss_by_W_depth_{index}']   = self._ssim_l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)

        return losses

    def _compute_val_metrics(self, depth_pred_uw, depth_pred_w, \
                                    rgb_uw, rgb_w, gt_depth_uw, gt_depth_w, mask_uw,
                                    conf_uw, conf_w, tof_uw_mask, tof_w_mask):

        metrics = dict()

        rgb_uw_pred_by_uw = self.Wwarp2UW_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_w)
        rgb_uw_pred_by_w = self.Wwarp2UW_by_W_depth(depth_pred_w*self.depth_scale, rgb_w)
        rgb_w_pred_by_w = self.UWwarp2W_by_W_depth(depth_pred_w*self.depth_scale, rgb_uw)
        rgb_w_pred_by_uw = self.UWwarp2W_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_uw)


        #metrics
        #UW depth
        mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
            gt_depth_uw/10 * conf_uw, depth_pred_uw * self.depth_scale /10 *conf_uw)

        metrics['depth_uw/mse'] = mse
        metrics['depth_uw/rmse'] = rmse
        metrics['depth_uw/mae'] = mae
        metrics['depth_uw/lg10'] = lg10
        metrics['depth_uw/absrel'] = absrel
        metrics['depth_uw/delta1'] = delta1
        metrics['depth_uw/delta2'] = delta2
        metrics['depth_uw/delta3'] = delta3
        metrics['depth_uw/sqrel'] = sqrel
        metrics['depth_uw/rmse_log'] = rmse_log

        # l1
        metrics['depth_uw/l1'] = F.l1_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale*conf_uw)
        # l2
        metrics['depth_uw/l2'] = F.mse_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale * conf_uw)
        # SSIM
        metrics['depth_uw/ssim'] = K.metrics.ssim(gt_depth_uw / self.depth_scale  * conf_uw, depth_pred_uw * conf_uw, 11).mean()
        # PSNR
        metrics['depth_uw/psnr'] = K.metrics.psnr(gt_depth_uw / self.depth_scale * conf_uw, depth_pred_uw * conf_uw, 1)

        if tof_uw_mask is not None:
            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_uw/10 * conf_uw*tof_uw_mask, depth_pred_uw * self.depth_scale /10 *conf_uw*tof_uw_mask)

            metrics['out_of_tof/depth_uw/mse'] = mse
            metrics['out_of_tof/depth_uw/rmse'] = rmse
            metrics['out_of_tof/depth_uw/mae'] = mae
            metrics['out_of_tof/depth_uw/lg10'] = lg10
            metrics['out_of_tof/depth_uw/absrel'] = absrel
            metrics['out_of_tof/depth_uw/delta1'] = delta1
            metrics['out_of_tof/depth_uw/delta2'] = delta2
            metrics['out_of_tof/depth_uw/delta3'] = delta3
            metrics['out_of_tof/depth_uw/sqrel'] = sqrel
            metrics['out_of_tof/depth_uw/rmse_log'] = rmse_log

            overlay_uw_mask = tof_uw_mask==False

            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_uw/10 * conf_uw*overlay_uw_mask, depth_pred_uw * self.depth_scale /10 *conf_uw*overlay_uw_mask)

            metrics['overlap_tof/depth_uw/mse'] = mse
            metrics['overlap_tof/depth_uw/rmse'] = rmse
            metrics['overlap_tof/depth_uw/mae'] = mae
            metrics['overlap_tof/depth_uw/lg10'] = lg10
            metrics['overlap_tof/depth_uw/absrel'] = absrel
            metrics['overlap_tof/depth_uw/delta1'] = delta1
            metrics['overlap_tof/depth_uw/delta2'] = delta2
            metrics['overlap_tof/depth_uw/delta3'] = delta3
            metrics['overlap_tof/depth_uw/sqrel'] = sqrel
            metrics['overlap_tof/depth_uw/rmse_log'] = rmse_log

        #W depth
        mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
            gt_depth_w/10 * conf_w, depth_pred_w * self.depth_scale/10 * conf_w)

        metrics['depth_w/mse'] = mse
        metrics['depth_w/rmse'] = rmse
        metrics['depth_w/mae'] = mae
        metrics['depth_w/lg10'] = lg10
        metrics['depth_w/absrel'] = absrel
        metrics['depth_w/delta1'] = delta1
        metrics['depth_w/delta2'] = delta2
        metrics['depth_w/delta3'] = delta3
        metrics['depth_w/sqrel'] = sqrel
        metrics['depth_w/rmse_log'] = rmse_log
        # l1
        metrics['depth_w/l1'] = F.l1_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w)
        # l2
        metrics['depth_w/l2'] = F.mse_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w)
        # SSIM
        metrics['depth_w/ssim'] = K.metrics.ssim(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 11).mean()
        # PSNR
        metrics['depth_w/psnr'] = K.metrics.psnr(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 1)

        if tof_w_mask is not None:
            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_w/10 * conf_w*tof_w_mask, depth_pred_w * self.depth_scale /10 *conf_w*tof_w_mask)
            
            metrics['out_of_tof/depth_w/mse'] = mse 
            metrics['out_of_tof/depth_w/rmse'] = rmse  
            metrics['out_of_tof/depth_w/mae'] = mae
            metrics['out_of_tof/depth_w/lg10'] = lg10
            metrics['out_of_tof/depth_w/absrel'] = absrel
            metrics['out_of_tof/depth_w/delta1'] = delta1
            metrics['out_of_tof/depth_w/delta2'] = delta2
            metrics['out_of_tof/depth_w/delta3'] = delta3
            metrics['out_of_tof/depth_w/sqrel'] = sqrel
            metrics['out_of_tof/depth_w/rmse_log'] = rmse_log

            overlay_w_mask = tof_w_mask==False

            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_w/10 * conf_w*overlay_w_mask, depth_pred_w * self.depth_scale /10 *conf_w*overlay_w_mask)
            
            metrics['overlap_tof/depth_w/mse'] = mse 
            metrics['overlap_tof/depth_w/rmse'] = rmse  
            metrics['overlap_tof/depth_w/mae'] = mae
            metrics['overlap_tof/depth_w/lg10'] = lg10
            metrics['overlap_tof/depth_w/absrel'] = absrel
            metrics['overlap_tof/depth_w/delta1'] = delta1
            metrics['overlap_tof/depth_w/delta2'] = delta2
            metrics['overlap_tof/depth_w/delta3'] = delta3
            metrics['overlap_tof/depth_w/sqrel'] = sqrel
            metrics['overlap_tof/depth_w/rmse_log'] = rmse_log

        #UW RGB
        #by UW
        # l1
        metrics['rgb_uw_by_uw_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw)
        # l2
        metrics['rgb_uw_by_uw_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw)
        # SSIM
        metrics['rgb_uw_by_uw_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 11).mean()
        # PSNR
        metrics['rgb_uw_by_uw_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 1)

        #by W
        # l1
        metrics['rgb_uw_by_w_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw)
        # l2
        metrics['rgb_uw_by_w_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw)
        # SSIM
        metrics['rgb_uw_by_w_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 11).mean()
        # PSNR
        metrics['rgb_uw_by_w_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 1)

        #W RGB
        #by W
        # l1
        metrics['rgb_w_by_w_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_w)
        # l2
        metrics['rgb_w_by_w_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_w)
        # SSIM
        metrics['rgb_w_by_w_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_w, 11).mean()
        # PSNR
        metrics['rgb_w_by_w_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_w, 1)
        
        #by UW
        # l1
        metrics['rgb_w_by_uw_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_uw)
        # l2
        metrics['rgb_w_by_uw_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_uw)
        # SSIM
        metrics['rgb_w_by_uw_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_uw, 11).mean()
        # PSNR
        metrics['rgb_w_by_uw_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_uw, 1)

        return metrics

    def _warp_train_data(self, gt_depth_uw, rgb_uw, ndepth_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw)
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        else:
            mask_uw = torch.ones_like(gt_depth_uw)
        
        uw_model_input = {'x_img':rgb_uw}

        w_model_input = {'x_img': rgb_w}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            uw_model_input['x_depth']= depth_uw
            w_model_input['x_depth']= depth_w

        train_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        if self.structure_loss:
            if self.struct_knowledge_src == 'midas':
                monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_uw = 1 - monodepth_uw
                monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_w = 1 - monodepth_w

                if self.resize4midas:
                    monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                    mode="bicubic")
                    monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                    mode="bicubic")

                train_loss_data['monodepth_uw'] = monodepth_uw
                train_loss_data['monodepth_w'] = monodepth_w

        return uw_model_input, w_model_input, train_loss_data

    def _epoch_fit(self, ):
        epoch_pbar = tqdm(iter(self.train_dataset))
        for gt_depth_uw, rgb_uw, ndepth_uw, conf in epoch_pbar:

            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None
            
            with torch.no_grad():
                uw_model_input, w_model_input, train_loss_data = \
                    self._warp_train_data(gt_depth_uw, rgb_uw, ndepth_uw)
                del gt_depth_uw, ndepth_uw

            depths_pred_uw = self.uw_model(**uw_model_input)
            depths_pred_w = self.w_model(**w_model_input)

            train_loss_data['depths_pred_uw'] = depths_pred_uw
            train_loss_data['depths_pred_w'] = depths_pred_w

            losses = self._compute_losses(**train_loss_data)

            total_loss = 0
            for loss_name in losses.keys():
                loss = losses[loss_name]
                weight = self.weights[loss_name]
                self.tb_writer.add_scalar('train_loss/{}'.format(loss_name), loss, self.n_iter)
                weighted_loss = loss * weight
                self.tb_writer.add_scalar('weighted_train_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
                total_loss += weighted_loss

            self.opt.zero_grad()
            if self.use_lr_decay:
                self._adjust_learning_rate()
            total_loss.backward()
            self.opt.step()

            self.tb_writer.add_scalar('train_loss/total_loss', total_loss, self.n_iter)

            self.n_iter+=1

    def _warp_val_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)

        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw)
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        else:
            mask_uw = torch.ones_like(gt_depth_uw)
        
        uw_model_input = {'x_img':rgb_uw}
        w_model_input = {'x_img': rgb_w}

        val_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        val_metric_data = {'rgb_uw':rgb_uw,
                            'rgb_w': rgb_w,
                            'gt_depth_uw': gt_depth_uw,
                            'gt_depth_w': gt_depth_w,
                            'mask_uw': mask_uw,
                            'conf_uw': conf_uw,
                            'conf_w' : conf_w,
                            'tof_uw_mask': None,
                            'tof_w_mask': None}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            uw_model_input['x_depth']= depth_uw
            w_model_input['x_depth']= depth_w
            tof_uw_mask = depth_uw==0
            tof_w_mask = depth_w==0
            val_metric_data['tof_uw_mask'] = tof_uw_mask
            val_metric_data['tof_w_mask'] = tof_w_mask

        if self.structure_loss:
            if self.struct_knowledge_src == 'midas':
                monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_uw = 1 - monodepth_uw
                monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_w = 1 - monodepth_w

                if self.resize4midas:
                    monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                    mode="bicubic")
                    monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                    mode="bicubic")

                val_loss_data['monodepth_uw'] = monodepth_uw
                val_loss_data['monodepth_w'] = monodepth_w

        return uw_model_input, w_model_input, val_loss_data, val_metric_data

    def _epoch_val(self,):
        print('epoch testing...')
        val_pbar = tqdm(iter(self.val_dataset))
        i=0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in val_pbar:
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            conf_uw = conf_uw.cuda()

            uw_model_input, w_model_input, val_loss_data, val_metric_data = \
                self._warp_val_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            depths_pred_uw = self.uw_model(**uw_model_input)
            depths_pred_w = self.w_model(**w_model_input)

            val_loss_data['depths_pred_uw'] = depths_pred_uw
            val_loss_data['depths_pred_w'] = depths_pred_w

            val_metric_data['depth_pred_uw'] = depths_pred_uw[-1]
            val_metric_data['depth_pred_w'] = depths_pred_w[-1]

            if i==0:
                total_losses = self._compute_losses(**val_loss_data)
    
                total_metrics = self._compute_val_metrics(**val_metric_data)

            else:
                losses = self._compute_losses(**val_loss_data)
                metrics = self._compute_val_metrics(**val_metric_data)

                for loss_name in losses.keys():
                    total_losses[loss_name] += losses[loss_name]

                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
            i+=1

        total_loss = 0
        weighted_total_loss = 0
        for loss_name in total_losses.keys():
            loss = total_losses[loss_name]/i
            self.tb_writer.add_scalar('val_loss/{}'.format(loss_name), loss, self.n_iter)
            total_loss+=loss 
            weighted_loss = loss * self.weights[loss_name]
            self.tb_writer.add_scalar('weighted_val_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
            weighted_total_loss += weighted_loss

        self.tb_writer.add_scalar('val_loss/total_loss', total_loss, self.n_iter)
        self.tb_writer.add_scalar('val_loss/weighted_total_loss', weighted_total_loss, self.n_iter)

        for metric_name in metrics.keys():
            metric = total_metrics[metric_name]/i
            self.tb_writer.add_scalar('metrics/{}'.format(metric_name), metric, self.n_iter)

            if metric_name == 'depth_uw/mae':
                if metric < self.uw_best_mae:
                    self.uw_best_mae_epoch = True
                    self.uw_best_mae = metric
                else:
                    self.uw_best_mae_epoch = False 

            if metric_name == 'depth_w/mae':
                if metric < self.w_best_mae:
                    self.w_best_mae_epoch = True
                    self.w_best_mae = metric
                else:
                    self.w_best_mae_epoch = False     

    def _epoch(self,):
        if self.fixed_img_encoder:
            self.uw_model.E_img.train(False)
            self.w_model.E_img.train(False)
            self.uw_model.D.train(True)
            self.w_model.D.train(True)
        else:
            self.uw_model.train(True)
            self.w_model.train(True)
        self._epoch_fit()
        self.uw_model.train(False)
        self.w_model.train(False)
        with torch.no_grad():
            self._visualize()
            self._epoch_val()
            if self.uw_best_mae_epoch:
                self._save_uw_model_cpt()
            if self.w_best_mae_epoch:
                self._save_w_model_cpt()
        self.epoch+=1

    def _visualize(self,):


        if 'RGBD' in self.mode:
            depths_pred_uw = self.uw_model(self.val_rgb_uw.cuda(), self.val_input_depth_uw.cuda())
            depths_pred_w = self.w_model(self.val_rgb_w.cuda(), self.val_input_depth_w.cuda())
        else:
            depths_pred_uw = self.uw_model(self.val_rgb_uw.cuda())
            depths_pred_w = self.w_model(self.val_rgb_w.cuda())

        save_path = os.path.join(self.img_dir, '{}'.format(self.n_iter))
        os.makedirs(save_path, exist_ok=True)

        for index in trange(5):
            depth_i_pred_uw = depths_pred_uw[index]
            depth_i_pred_w = depths_pred_w[index]
            
            rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_w.cuda())
            rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_w.cuda())
            rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_uw.cuda())
            rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_uw.cuda())

            self.tb_writer.add_images(f'test/depth_uw_{index}', depth_i_pred_uw, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw, os.path.join(save_path, f'depth_pred_uw_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_uw', depth_i_pred_uw, save_path) 

            depth_i_pred_uw_conf = depth_i_pred_uw * self.val_conf_uw.cuda()
            self.tb_writer.add_images(f'test/depth_uw_{index}_conf', depth_i_pred_uw_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw_conf, os.path.join(save_path, f'depth_pred_uw_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_uw_conf', depth_i_pred_uw_conf, save_path) 

            self.tb_writer.add_images(f'test/depth_w_{index}', depth_i_pred_w, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_w, os.path.join(save_path, f'depth_pred_w_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_w', depth_i_pred_w, save_path) 
            
            depth_i_pred_w_conf = depth_i_pred_w * self.val_conf_w.cuda()
            self.tb_writer.add_images(f'test/depth_w_{index}_conf', depth_i_pred_w_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_w_conf, os.path.join(save_path, f'depth_pred_w_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_w_conf', depth_i_pred_w_conf, save_path) 

            self.tb_writer.add_images(f'val_rgb/rgb_uw_pred_by_depth_{index}_uw', rgb_uw_pred_by_depth_i_uw, self.n_iter)
            torchvision.utils.save_image(rgb_uw_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_uw.png'))  
            
            self.tb_writer.add_images(f'val_rgb/uw_pred_by_depth_{index}_w', rgb_uw_pred_by_depth_i_w, self.n_iter)
            torchvision.utils.save_image(rgb_uw_pred_by_depth_i_w, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_w.png'))  

            self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_w', rgb_w_pred_by_depth_i_w, self.n_iter)
            torchvision.utils.save_image(rgb_w_pred_by_depth_i_w, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_w.png'))  
            
            self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_uw', rgb_w_pred_by_depth_i_uw, self.n_iter)
            torchvision.utils.save_image(rgb_w_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_uw.png')) 

    def _save_model_cpt(self,):
        self._save_uw_model_cpt()
        self._save_w_model_cpt()

    def _save_uw_model_cpt(self,):
        save_path = os.path.join(self.cpt_dir, 'UW_E{}_iter_{}.cpt'.format(self.epoch, self.n_iter))
        print('saving uw model cpt @ {}'.format(save_path))
        print('UW MAE(cm): {}'.format(self.uw_best_mae))
        torch.save(self.uw_model.state_dict(), save_path)

    def _save_w_model_cpt(self,):
        save_path = os.path.join(self.cpt_dir, 'W_E{}_iter_{}.cpt'.format(self.epoch, self.n_iter))
        print('saving w model cpt @ {}'.format(save_path))
        print('W MAE(cm): {}'.format(self.w_best_mae))
        torch.save(self.w_model.state_dict(), save_path)

class TFT3D_SingleSplitModelTrainer(BasicTrainer):
    def __init__(self, model, model_cam):
        super().__init__()
        self.model  = model.cuda() 
        self.model_cam  = model_cam

        self.n_iter = 0
        self.epoch = 0

    def init_optim(self, lr, betas, lr_decay, use_radam, use_lr_decay,
                   fixed_img_encoder ):
        print('init optimizer....')
        self.lr = lr 
        self.betas = betas
        self.use_radam = use_radam
        self.lr_decay = lr_decay
        self.use_lr_decay = use_lr_decay

        self.fixed_img_encoder = fixed_img_encoder

        update_models = []
        if fixed_img_encoder:
            update_models.append(self.model.D)
        else:
            update_models.append(self.model)

        update_models = nn.ModuleList(update_models)

        if self.use_radam:
            self.opt = torch.optim.RAdam(update_models.parameters(), lr=self.lr, betas=self.betas)
        else:
            self.opt = torch.optim.Adam(update_models.parameters(), lr=self.lr, betas=self.betas)


    def init_val_data(self,):
        gt_depth_uw, rgb_uw, ndepth_uw, conf_uw = next(iter(self.val_dataset))
        gt_depth_uw = gt_depth_uw.cuda()
        rgb_uw = rgb_uw.cuda()
        if 'RGBD' in self.mode:
            ndepth_uw = ndepth_uw.cuda()
        else:
            ndepth_uw = None
        conf_uw = conf_uw.cuda()


        tof_depth = None
        self.val_conf_uw = None  
        self.val_rgb_uw = None
        self.val_conf_w = None
        self.val_rgb_w = None
        self.val_input_depth_uw = None
        self.val_input_depth_w = None
        input_depth_uw = None
        input_depth_w = None

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)
        self.val_rgb_w = rgb_w.cpu()
        self.val_rgb_uw = rgb_uw.cpu()

        if self.model_cam == 'w':
            gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
            conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)
            self.val_conf_w = conf_w.cpu()
        if self.model_cam == 'uw':
            self.val_conf_uw = conf_uw.cpu()          

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            if self.model_cam == 'uw':
                input_depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                self.val_input_depth_uw = input_depth_uw.cpu()
            if self.model_cam == 'w':
                input_depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                self.val_input_depth_w = input_depth_w.cpu()


        save_path = os.path.join(self.img_dir, 'ground_truth')
        os.makedirs(save_path, exist_ok=True)

        if tof_depth is not None:
            tof_depth /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/tof_depth', tof_depth, self.n_iter)
            torchvision.utils.save_image(tof_depth, os.path.join(save_path, f'tof_depth.png')) 
            self._visualize_depths('ground_truth/tof_depth', tof_depth, save_path)


        if self.val_input_depth_uw is not None:
            self.tb_writer.add_images(f'input/depth_uw', self.val_input_depth_uw, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_uw, os.path.join(save_path, f'input_depth_uw.png')) 
            self._visualize_depths('input/depth_uw', self.val_input_depth_uw, save_path)
        if self.val_input_depth_w is not None:
            self.tb_writer.add_images(f'input/depth_w', self.val_input_depth_w, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_w, os.path.join(save_path, f'input_depth_w.png'))
            self._visualize_depths('input/depth_w', self.val_input_depth_w, save_path)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw) * self.depth_scale
            mask_uw = self.Wwarp2UW_by_UW_depth(mask_uw, mask_uw)>0
            mask_uw = mask_uw.float()
            self.tb_writer.add_images(f'input/mask_uw', mask_uw, self.n_iter)
            torchvision.utils.save_image(mask_uw, os.path.join(save_path, f'mask_uw.png'))  

        if self.model_cam == 'uw':

            self.tb_writer.add_images(f'ground_truth/rgb_uw', rgb_uw, self.n_iter)
            torchvision.utils.save_image(rgb_uw, os.path.join(save_path, f'rgb_uw.png'))
            gt_depth_uw /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/gt_depth_uw', gt_depth_uw, self.n_iter)
            torchvision.utils.save_image(gt_depth_uw, os.path.join(save_path, f'gt_depth_uw.png')) 
            self._visualize_depths('gt_depth_uw/depth_uw', gt_depth_uw, save_path)

            gt_depth_uw_conf = gt_depth_uw * conf_uw
            self.tb_writer.add_images(f'ground_truth/gt_depth_uw_conf', gt_depth_uw_conf, self.n_iter)
            torchvision.utils.save_image(gt_depth_uw_conf, os.path.join(save_path, f'gt_depth_uw_conf.png')) 
            self._visualize_depths('gt_depth_uw/depth_uw_conf', gt_depth_uw_conf, save_path)

        if self.model_cam == 'w':
            self.tb_writer.add_images(f'ground_truth/rgb_w', rgb_w, self.n_iter)
            torchvision.utils.save_image(rgb_w, os.path.join(save_path, f'rgb_w.png'))     
            gt_depth_w /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/gt_depth_w', gt_depth_w, self.n_iter)
            torchvision.utils.save_image(gt_depth_w, os.path.join(save_path, f'gt_depth_w.png')) 
            self._visualize_depths('gt_depth_w/depth_w', gt_depth_w, save_path)

            gt_depth_w_conf = gt_depth_w * conf_w
            self.tb_writer.add_images(f'ground_truth/gt_depth_w_conf', gt_depth_w_conf, self.n_iter)
            torchvision.utils.save_image(gt_depth_w_conf, os.path.join(save_path, f'gt_depth_w.png')) 
            self._visualize_depths('gt_depth_w/depth_w_conf', gt_depth_w, save_path)

    def _compute_losses(self, depths_pred_uw, depths_pred_w, \
                        rgb_uw, rgb_w, monodepth_uw, monodepth_w, mask_uw):
        losses = dict()

        for index in range(5):
            if self.model_cam == 'uw':
                depth_i_pred_uw = depths_pred_uw[index]
                rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, rgb_w)
                rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, rgb_uw)
                if self.photo_loss =='l1':
                    losses[f'UW_l1_warp_loss_by_UW_depth_{index}'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'W_l1_warp_loss_by_UW_depth_{index}'] = F.l1_loss(rgb_w,rgb_w_pred_by_depth_i_uw)
                elif self.photo_loss =='l2':
                    losses[f'UW_l2_warp_loss_by_UW_depth_{index}'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'W_l2_warp_loss_by_UW_depth_{index}'] = F.mse_loss(rgb_w,rgb_w_pred_by_depth_i_uw)
                elif self.photo_loss == 'ssim':
                    losses[f'UW_ssim_warp_loss_by_UW_depth_{index}'] = self._ssim_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw).mean()
                    losses[f'W_ssim_warp_loss_by_UW_depth_{index}'] = self._ssim_loss(rgb_w,rgb_w_pred_by_depth_i_uw).mean()
                elif self.photo_loss == 'ssim_l1':
                    losses[f'UW_ssim_l1_warp_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw).mean()
                    losses[f'W_ssim_l1_warp_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(rgb_w,rgb_w_pred_by_depth_i_uw).mean()
                if self.smooth_loss:
                    losses[f'UW_smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_uw, rgb_uw)
                if self.structure_loss:
                    losses[f'UW_structure_ssim_loss_{index}'] = self._ssim_loss(self._normalize_depth(depth_i_pred_uw), self._normalize_depth(monodepth_uw)).mean()

            if self.model_cam == 'w':
                depth_i_pred_w = depths_pred_w[index]
                rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, rgb_w)
                rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, rgb_uw)
                if self.photo_loss =='l1':
                    losses[f'UW_l1_warp_loss_by_W_depth_{index}'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_l1_warp_loss_by_W_depth_{index}'] = F.l1_loss(rgb_w,rgb_w_pred_by_depth_i_w)
                elif self.photo_loss =='l2':
                    losses[f'UW_l2_warp_loss_by_W_depth_{index}'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_l2_warp_loss_by_W_depth_{index}'] = F.mse_loss(rgb_w,rgb_w_pred_by_depth_i_w)
                elif self.photo_loss == 'ssim':
                    losses[f'UW_ssim_warp_loss_by_W_depth_{index}'] = self._ssim_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw).mean()
                    losses[f'W_ssim_warp_loss_by_W_depth_{index}'] = self._ssim_loss(rgb_w,rgb_w_pred_by_depth_i_w).mean()
                elif self.photo_loss == 'ssim_l1':
                    losses[f'UW_ssim_l1_warp_loss_by_W_depth_{index}'] = self._ssim_l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw).mean()
                    losses[f'W_ssim_l1_warp_loss_by_W_depth_{index}'] = self._ssim_l1_loss(rgb_w,rgb_w_pred_by_depth_i_w).mean()

                if self.smooth_loss:
                    losses[f'W_smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_w, rgb_w)
                if self.structure_loss:
                    losses[f'W_structure_ssim_loss_{index}']  = self._ssim_loss(self._normalize_depth(depth_i_pred_w), self._normalize_depth(monodepth_w)).mean()

        return losses

    def _compute_val_metrics(self, depth_pred_uw, depth_pred_w, \
                                    rgb_uw, rgb_w, gt_depth_uw, gt_depth_w, mask_uw,
                                    conf_uw, conf_w, tof_uw_mask=None, tof_w_mask=None):

        metrics = dict()

        if self.model_cam == 'uw':
            rgb_uw_pred_by_uw = self.Wwarp2UW_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_w)
            rgb_w_pred_by_uw = self.UWwarp2W_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_uw)
        if self.model_cam == 'w':
            rgb_uw_pred_by_w = self.Wwarp2UW_by_W_depth(depth_pred_w*self.depth_scale, rgb_w)
            rgb_w_pred_by_w = self.UWwarp2W_by_W_depth(depth_pred_w*self.depth_scale, rgb_uw)


        #metrics
        #UW depth
        if self.model_cam == 'uw':
            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_uw/10 * conf_uw, depth_pred_uw * self.depth_scale /10 *conf_uw)

            metrics['depth_uw/mse'] = mse
            metrics['depth_uw/rmse'] = rmse
            metrics['depth_uw/mae'] = mae
            metrics['depth_uw/lg10'] = lg10
            metrics['depth_uw/absrel'] = absrel
            metrics['depth_uw/delta1'] = delta1
            metrics['depth_uw/delta2'] = delta2
            metrics['depth_uw/delta3'] = delta3
            metrics['depth_uw/sqrel'] = sqrel
            metrics['depth_uw/rmse_log'] = rmse_log

            # l1
            metrics['depth_uw/l1'] = F.l1_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale*conf_uw)
            # l2
            metrics['depth_uw/l2'] = F.mse_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale * conf_uw)
            # SSIM
            metrics['depth_uw/ssim'] = K.metrics.ssim(gt_depth_uw / self.depth_scale  * conf_uw, depth_pred_uw * conf_uw, 11).mean()
            # PSNR
            metrics['depth_uw/psnr'] = K.metrics.psnr(gt_depth_uw / self.depth_scale * conf_uw, depth_pred_uw * conf_uw, 1)

            if tof_uw_mask is not None:
                mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                    gt_depth_uw/10 * conf_uw*tof_uw_mask, depth_pred_uw * self.depth_scale /10 *conf_uw*tof_uw_mask)

                metrics['our_of_tof/depth_uw/mse'] = mse
                metrics['our_of_tof/depth_uw/rmse'] = rmse
                metrics['our_of_tof/depth_uw/mae'] = mae
                metrics['our_of_tof/depth_uw/lg10'] = lg10
                metrics['our_of_tof/depth_uw/absrel'] = absrel
                metrics['our_of_tof/depth_uw/delta1'] = delta1
                metrics['our_of_tof/depth_uw/delta2'] = delta2
                metrics['our_of_tof/depth_uw/delta3'] = delta3
                metrics['our_of_tof/depth_uw/sqrel'] = sqrel
                metrics['our_of_tof/depth_uw/rmse_log'] = rmse_log

        #W depth
        if self.model_cam == 'w':
            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_w/10 * conf_w, depth_pred_w * self.depth_scale/10 * conf_w)

            metrics['depth_w/mse'] = mse
            metrics['depth_w/rmse'] = rmse
            metrics['depth_w/mae'] = mae
            metrics['depth_w/lg10'] = lg10
            metrics['depth_w/absrel'] = absrel
            metrics['depth_w/delta1'] = delta1
            metrics['depth_w/delta2'] = delta2
            metrics['depth_w/delta3'] = delta3
            metrics['depth_w/sqrel'] = sqrel
            metrics['depth_w/rmse_log'] = rmse_log
            # l1
            metrics['depth_w/l1'] = F.l1_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w)
            # l2
            metrics['depth_w/l2'] = F.mse_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w)
            # SSIM
            metrics['depth_w/ssim'] = K.metrics.ssim(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 11).mean()
            # PSNR
            metrics['depth_w/psnr'] = K.metrics.psnr(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 1)

            if tof_w_mask is not None:
                mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                    gt_depth_w/10 * conf_w*tof_w_mask, depth_pred_w * self.depth_scale /10 *conf_w*tof_w_mask)
                
                metrics['our_of_tof/depth_w/mse'] = mse
                metrics['our_of_tof/depth_w/rmse'] = rmse  
                metrics['our_of_tof/depth_w/mae'] = mae
                metrics['our_of_tof/depth_w/lg10'] = lg10
                metrics['our_of_tof/depth_w/absrel'] = absrel
                metrics['our_of_tof/depth_w/delta1'] = delta1
                metrics['our_of_tof/depth_w/delta2'] = delta2
                metrics['our_of_tof/depth_w/delta3'] = delta3
                metrics['our_of_tof/depth_w/sqrel'] = sqrel
                metrics['our_of_tof/depth_w/rmse_log'] = rmse_log

        # by UW 
        if self.model_cam == 'uw':
            #UW RGB
            # l1
            metrics['rgb_uw_by_uw_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw)
            # l2
            metrics['rgb_uw_by_uw_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw)
            # SSIM
            metrics['rgb_uw_by_uw_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 11).mean()
            # PSNR
            metrics['rgb_uw_by_uw_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 1)
            # W RGB
            # l1
            metrics['rgb_w_by_uw_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_uw)
            # l2
            metrics['rgb_w_by_uw_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_uw)
            # SSIM
            metrics['rgb_w_by_uw_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_uw, 11).mean()
            # PSNR
            metrics['rgb_w_by_uw_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_uw, 1)
        # by W
        if self.model_cam == 'w':
            #UW RGB
            #by W
            # l1
            metrics['rgb_uw_by_w_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw)
            # l2
            metrics['rgb_uw_by_w_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw)
            # SSIM
            metrics['rgb_uw_by_w_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 11).mean()
            # PSNR
            metrics['rgb_uw_by_w_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 1)

            #W RGB
            #by W
            # l1
            metrics['rgb_w_by_w_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_w)
            # l2
            metrics['rgb_w_by_w_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_w)
            # SSIM
            metrics['rgb_w_by_w_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_w, 11).mean()
            # PSNR
            metrics['rgb_w_by_w_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_w, 1)
        
        return metrics

    def _warp_train_data(self, gt_depth_uw, rgb_uw, ndepth_uw):

        
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        mask_uw = torch.ones_like(gt_depth_uw)
        if self.use_mask:
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        
        if self.model_cam == 'uw':
            model_input = {'x_img':rgb_uw}

        if self.model_cam == 'w':
            model_input = {'x_img': rgb_w}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            if self.model_cam == 'uw':
                depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth']= depth_uw
            if self.model_cam == 'w':
                depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth']= depth_w

        train_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        if self.structure_loss:
            if self.model_cam == 'uw':
                monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_uw = 1 - monodepth_uw
                if self.resize4midas:
                    monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                    mode="bicubic")
                train_loss_data['monodepth_uw'] = monodepth_uw
            if self.model_cam == 'w':
                monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_w = 1 - monodepth_w
                if self.resize4midas:
                    monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                    mode="bicubic")
                train_loss_data['monodepth_w'] = monodepth_w

        return model_input, train_loss_data

    def _epoch_fit(self, ):
        epoch_pbar = tqdm(iter(self.train_dataset))
        for gt_depth_uw, rgb_uw, ndepth_uw, conf in epoch_pbar:

            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            with torch.no_grad():
                model_input, train_loss_data = \
                    self._warp_train_data(gt_depth_uw, rgb_uw, ndepth_uw)
                del gt_depth_uw, ndepth_uw

            depths_pred = self.model(**model_input)

            if self.model_cam == 'uw':
                train_loss_data['depths_pred_uw'] = depths_pred
            else:
                train_loss_data['depths_pred_uw'] = None
            if self.model_cam == 'w':
                train_loss_data['depths_pred_w'] = depths_pred
            else:
                train_loss_data['depths_pred_w'] = None

            losses = self._compute_losses(**train_loss_data)

            total_loss = 0
            for loss_name in losses.keys():
                loss = losses[loss_name]
                weight = self.weights[loss_name]
                self.tb_writer.add_scalar('train_loss/{}'.format(loss_name), loss, self.n_iter)
                weighted_loss = loss * weight
                self.tb_writer.add_scalar('weighted_train_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
                total_loss += weighted_loss

            self.opt.zero_grad()
            if self.use_lr_decay:
                self._adjust_learning_rate()
            total_loss.backward()
            self.opt.step()

            self.tb_writer.add_scalar('train_loss/total_loss', total_loss, self.n_iter)

            self.n_iter+=1

    def _warp_val_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        mask_uw = torch.ones_like(gt_depth_uw)

        gt_depth_w = None
        depth_w = None
        conf_w = None

        if self.use_mask:
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)
        if self.model_cam == 'uw':
            model_input = {'x_img':rgb_uw}
        if self.model_cam == 'w':
            gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
            conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)
            model_input = {'x_img': rgb_w}
            

        val_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        val_metric_data = {'rgb_uw':rgb_uw,
                            'rgb_w': rgb_w,
                            'gt_depth_uw': gt_depth_uw,
                            'gt_depth_w': gt_depth_w,
                            'mask_uw': mask_uw,
                            'conf_uw': conf_uw,
                            'conf_w' : conf_w,
                            'tof_uw_mask': None,
                            'tof_w_mask': None,}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            if self.model_cam == 'uw':
                depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth'] = depth_uw
                tof_uw_mask = depth_uw==0
                val_metric_data['tof_uw_mask'] = tof_uw_mask
            if self.model_cam == 'w':
                depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth'] = depth_w
                tof_w_mask = depth_w==0
                val_metric_data['tof_w_mask'] = tof_w_mask

        if self.structure_loss:
            if self.model_cam == 'uw':
                monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_uw = 1 - monodepth_uw
                if self.resize4midas:
                    monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth.shape[2:],
                                                    mode="bicubic")
                val_loss_data['monodepth_uw'] = monodepth_uw
            if self.model_cam == 'w':
                monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_w = 1 - monodepth_w
                if self.resize4midas:
                    monodepth_w = F.interpolate(monodepth_w, size=gt_depth.shape[2:],
                                                    mode="bicubic")
                val_loss_data['monodepth_w'] = monodepth_w

        return model_input, val_loss_data, val_metric_data

    def _epoch_val(self,):
        print('epoch testing...')
        val_pbar = tqdm(iter(self.val_dataset))
        i=0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in val_pbar:
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            conf_uw = conf_uw.cuda()

            model_input, val_loss_data, val_metric_data = \
                self._warp_val_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            depths_pred = self.model(**model_input)

            if self.model_cam == 'uw':
                val_loss_data['depths_pred_uw'] = depths_pred
                val_metric_data['depth_pred_uw'] = depths_pred[-1]
            else:
                val_loss_data['depths_pred_uw'] = None
                val_metric_data['depth_pred_uw'] = None
            if self.model_cam == 'w':
                val_loss_data['depths_pred_w'] = depths_pred
                val_metric_data['depth_pred_w'] = depths_pred[-1]
            else:
                val_loss_data['depths_pred_w'] = None
                val_metric_data['depth_pred_w'] = None

            if i==0:
                total_losses = self._compute_losses(**val_loss_data)
    
                total_metrics = self._compute_val_metrics(**val_metric_data)

            else:
                losses = self._compute_losses(**val_loss_data)
                metrics = self._compute_val_metrics(**val_metric_data)

                for loss_name in losses.keys():
                    total_losses[loss_name] += losses[loss_name]

                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
            i+=1

        total_loss = 0
        weighted_total_loss = 0
        for loss_name in total_losses.keys():
            loss = total_losses[loss_name]/i
            self.tb_writer.add_scalar('val_loss/{}'.format(loss_name), loss, self.n_iter)
            total_loss+=loss 
            weighted_loss = loss * self.weights[loss_name]
            self.tb_writer.add_scalar('weighted_val_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
            weighted_total_loss += weighted_loss

        self.tb_writer.add_scalar('val_loss/total_loss', total_loss, self.n_iter)
        self.tb_writer.add_scalar('val_loss/weighted_total_loss', weighted_total_loss, self.n_iter)

        for metric_name in metrics.keys():
            metric = total_metrics[metric_name]/i
            self.tb_writer.add_scalar('metrics/{}'.format(metric_name), metric, self.n_iter)

            if metric_name == 'depth_uw/mae':
                if metric < self.uw_best_mae:
                    self.uw_best_mae_epoch = True
                    self.uw_best_mae = metric
                else:
                    self.uw_best_mae_epoch = False 

            if metric_name == 'depth_w/mae':
                if metric < self.w_best_mae:
                    self.w_best_mae_epoch = True
                    self.w_best_mae = metric
                else:
                    self.w_best_mae_epoch = False     

    def _epoch(self,):
        if self.fixed_img_encoder:
            self.model.E_img.train(False)
            self.model.D.train(True)
        else:
            self.model.train(True)
        self._epoch_fit()
        self.model.train(False)
        with torch.no_grad():
            self._visualize()
            self._epoch_val()
            if self.uw_best_mae_epoch:
                self._save_uw_model_cpt()
            if self.w_best_mae_epoch:
                self._save_w_model_cpt()
        self.epoch+=1

    def _visualize(self,):
        
        if self.model_cam == 'uw':
            if self.val_input_depth_uw is not None:
                depths_pred_uw = self.model(self.val_rgb_uw.cuda(), self.val_input_depth_uw.cuda())
            else:
                depths_pred_uw = self.model(self.val_rgb_uw.cuda())

        if self.model_cam == 'w':
            if self.val_input_depth_w is not None:
                depths_pred_w = self.model(self.val_rgb_w.cuda(), self.val_input_depth_w.cuda())
            else:
                depths_pred_w = self.model(self.val_rgb_w.cuda())

        save_path = os.path.join(self.img_dir, '{}'.format(self.n_iter))
        os.makedirs(save_path, exist_ok=True)

        for index in trange(5):
            if self.model_cam == 'uw':
                depth_i_pred_uw = depths_pred_uw[index]
                rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_w.cuda())
                rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_uw.cuda())
                
                self.tb_writer.add_images(f'test/depth_uw_{index}', depth_i_pred_uw, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_uw, os.path.join(save_path, f'depth_pred_uw_{index}.png'))
                self._visualize_depths(f'val_depth_{index}/depth_uw', depth_i_pred_uw, save_path) 
                
                depth_i_pred_uw_conf = depth_i_pred_uw * self.val_conf_uw.cuda()
                self.tb_writer.add_images(f'test/depth_uw_{index}_conf', depth_i_pred_uw_conf, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_uw_conf, os.path.join(save_path, f'depth_pred_uw_{index}_conf.png'))
                self._visualize_depths(f'val_depth_{index}/depth_uw_conf', depth_i_pred_uw_conf, save_path) 

                self.tb_writer.add_images(f'val_rgb/rgb_uw_pred_by_depth_{index}_uw', rgb_uw_pred_by_depth_i_uw, self.n_iter)
                torchvision.utils.save_image(rgb_uw_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_uw.png'))  
                
                self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_uw', rgb_w_pred_by_depth_i_uw, self.n_iter)
                torchvision.utils.save_image(rgb_w_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_uw.png')) 

            if self.model_cam == 'w':
                depth_i_pred_w = depths_pred_w[index]
                rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_w.cuda())
                rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_uw.cuda())

                self.tb_writer.add_images(f'test/depth_w_{index}', depth_i_pred_w, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_w, os.path.join(save_path, f'depth_pred_w_{index}.png'))
                self._visualize_depths(f'val_depth_{index}/depth_w', depth_i_pred_w, save_path) 
                
                depth_i_pred_w_conf = depth_i_pred_w * self.val_conf_w.cuda()
                self.tb_writer.add_images(f'test/depth_w_{index}_conf', depth_i_pred_w_conf, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_w_conf, os.path.join(save_path, f'depth_pred_w_{index}_conf.png'))
                self._visualize_depths(f'val_depth_{index}/depth_w_conf', depth_i_pred_w_conf, save_path) 
                
                self.tb_writer.add_images(f'val_rgb/uw_pred_by_depth_{index}_w', rgb_uw_pred_by_depth_i_w, self.n_iter)
                torchvision.utils.save_image(rgb_uw_pred_by_depth_i_w, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_w.png'))  

                self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_w', rgb_w_pred_by_depth_i_w, self.n_iter)
                torchvision.utils.save_image(rgb_w_pred_by_depth_i_w, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_w.png'))  

    def _save_model_cpt(self,):
        if self.model_cam == 'uw':
            self._save_uw_model_cpt()
        if self.model_cam == 'w':
            self._save_w_model_cpt()

    def _save_uw_model_cpt(self,):
        save_path = os.path.join(self.cpt_dir, 'UW_E{}_iter_{}.cpt'.format(self.epoch, self.n_iter))
        print('saving uw model cpt @ {}'.format(save_path))
        print('UW MAE(cm): {}'.format(self.uw_best_mae))
        torch.save(self.model.state_dict(), save_path)

    def _save_w_model_cpt(self,):
        save_path = os.path.join(self.cpt_dir, 'W_E{}_iter_{}.cpt'.format(self.epoch, self.n_iter))
        print('saving w model cpt @ {}'.format(save_path))
        print('W MAE(cm): {}'.format(self.w_best_mae))
        torch.save(self.model.state_dict(), save_path)

class TFT3D_StereoModelTrainer(BasicTrainer):

    def __init__(self,model):
        super().__init__()
        self.model  = model.cuda() 

        self.n_iter = 0
        self.epoch = 0

    def init_optim(self, lr, betas, lr_decay, use_radam, use_lr_decay,
                   fixed_img_encoder ):
        print('init optimizer....')
        self.lr = lr 
        self.betas = betas
        self.use_radam = use_radam
        self.lr_decay = lr_decay
        self.use_lr_decay = use_lr_decay

        self.fixed_img_encoder = fixed_img_encoder

        update_models = []
        if fixed_img_encoder:
            update_models.append(self.model.D)
        else:
            update_models.append(self.model)

        update_models = nn.ModuleList(update_models)

        if self.use_radam:
            self.opt = torch.optim.RAdam(update_models.parameters(), lr=self.lr, betas=self.betas)
        else:
            self.opt = torch.optim.Adam(update_models.parameters(), lr=self.lr, betas=self.betas)


    def init_val_data(self,):
        gt_depth_uw, rgb_uw, ndepth_uw, conf_uw = next(iter(self.val_dataset))
        gt_depth_uw = gt_depth_uw.cuda()
        rgb_uw = rgb_uw.cuda()
        ndepth_uw = ndepth_uw.cuda()
        conf_uw = conf_uw.cuda()
        self.val_conf_uw = conf_uw.cpu()

        tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        self.val_rgb_uw = rgb_uw.cpu()
        self.val_rgb_w = rgb_w.cpu()
        self.val_input_depth_uw = None
        self.val_input_depth_w = None
            
        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)
        self.val_conf_w = conf_w.cpu()

        save_path = os.path.join(self.img_dir, 'ground_truth')
        os.makedirs(save_path, exist_ok=True)

        tof_depth /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/tof_depth', tof_depth, self.n_iter)
        torchvision.utils.save_image(tof_depth, os.path.join(save_path, f'tof_depth.png')) 
        self._visualize_depths('ground_truth/tof_depth', tof_depth, save_path)


        if self.val_input_depth_uw:
            self.tb_writer.add_images(f'input/depth_uw', self.val_input_depth_uw, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_uw, os.path.join(save_path, f'input_depth_uw.png')) 
            self._visualize_depths('input/depth_uw', self.val_input_depth_uw, save_path)
        if self.val_input_depth_w:
            self.tb_writer.add_images(f'input/depth_w', self.val_input_depth_w, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_w, os.path.join(save_path, f'input_depth_w.png'))
            self._visualize_depths('input/depth_w', self.val_input_depth_w, save_path)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw) * self.depth_scale
            mask_uw = self.Wwarp2UW_by_UW_depth(mask_uw, mask_uw)>0
            mask_uw = mask_uw.float()
            self.tb_writer.add_images(f'input/mask_uw', mask_uw, self.n_iter)
            torchvision.utils.save_image(mask_uw, os.path.join(save_path, f'mask_uw.png'))

        self.tb_writer.add_images(f'ground_truth/rgb_uw', rgb_uw, self.n_iter)
        torchvision.utils.save_image(rgb_uw, os.path.join(save_path, f'rgb_uw.png')) 
        self.tb_writer.add_images(f'ground_truth/rgb_w', rgb_w, self.n_iter)
        torchvision.utils.save_image(rgb_w, os.path.join(save_path, f'rgb_w.png'))     

        gt_depth_uw /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/gt_depth_uw', gt_depth_uw, self.n_iter)
        torchvision.utils.save_image(gt_depth_uw, os.path.join(save_path, f'gt_depth_uw.png')) 
        self._visualize_depths('gt_depth_uw/depth_uw', gt_depth_uw, save_path)

        gt_depth_uw_conf = gt_depth_uw * conf_uw
        self.tb_writer.add_images(f'ground_truth/gt_depth_uw_conf', gt_depth_uw_conf, self.n_iter)
        torchvision.utils.save_image(gt_depth_uw_conf, os.path.join(save_path, f'gt_depth_uw_conf.png')) 
        self._visualize_depths('gt_depth_uw/depth_uw_conf', gt_depth_uw_conf, save_path)

        gt_depth_w /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/gt_depth_w', gt_depth_w, self.n_iter)
        torchvision.utils.save_image(gt_depth_w, os.path.join(save_path, f'gt_depth_w.png')) 
        self._visualize_depths('gt_depth_w/depth_w', gt_depth_w, save_path)

        gt_depth_w_conf = gt_depth_w * conf_w
        self.tb_writer.add_images(f'ground_truth/gt_depth_w_conf', gt_depth_w_conf, self.n_iter)
        torchvision.utils.save_image(gt_depth_w_conf, os.path.join(save_path, f'gt_depth_w.png')) 
        self._visualize_depths('gt_depth_w/depth_w_conf', gt_depth_w, save_path)


    def _compute_val_metrics(self, depth_pred_uw, depth_pred_w, \
                                    rgb_uw, rgb_w, gt_depth_uw, gt_depth_w, mask_uw,
                                    conf_uw, conf_w):

        metrics = dict()

        rgb_uw_pred_by_uw = self.Wwarp2UW_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_w)
        rgb_w_pred_by_uw = self.UWwarp2W_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_uw)
        rgb_uw_pred_by_w = self.Wwarp2UW_by_W_depth(depth_pred_w*self.depth_scale, rgb_w)
        rgb_w_pred_by_w = self.UWwarp2W_by_W_depth(depth_pred_w*self.depth_scale, rgb_uw)

        #metrics
        #UW depth
        mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
            gt_depth_uw/10 * conf_uw, depth_pred_uw * self.depth_scale /10 *conf_uw)

        metrics['depth_uw/mse'] = mse
        metrics['depth_uw/rmse'] = rmse
        metrics['depth_uw/mae'] = mae
        metrics['depth_uw/lg10'] = lg10
        metrics['depth_uw/absrel'] = absrel
        metrics['depth_uw/delta1'] = delta1
        metrics['depth_uw/delta2'] = delta2
        metrics['depth_uw/delta3'] = delta3
        metrics['depth_uw/sqrel'] = sqrel
        metrics['depth_uw/rmse_log'] = rmse_log

        # l1
        metrics['depth_uw/l1'] = F.l1_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale*conf_uw)
        # l2
        metrics['depth_uw/l2'] = F.mse_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale * conf_uw)
        # SSIM
        metrics['depth_uw/ssim'] = K.metrics.ssim(gt_depth_uw / self.depth_scale  * conf_uw, depth_pred_uw * conf_uw, 11).mean()
        # PSNR
        metrics['depth_uw/psnr'] = K.metrics.psnr(gt_depth_uw / self.depth_scale * conf_uw, depth_pred_uw * conf_uw, 1)

        #W depth
        mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
            gt_depth_w/10 * conf_w, depth_pred_w * self.depth_scale/10 * conf_w)

        metrics['depth_w/mse'] = mse
        metrics['depth_w/rmse'] = rmse
        metrics['depth_w/mae'] = mae
        metrics['depth_w/lg10'] = lg10
        metrics['depth_w/absrel'] = absrel
        metrics['depth_w/delta1'] = delta1
        metrics['depth_w/delta2'] = delta2
        metrics['depth_w/delta3'] = delta3
        metrics['depth_w/sqrel'] = sqrel
        metrics['depth_w/rmse_log'] = rmse_log
        # l1
        metrics['depth_w/l1'] = F.l1_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w)
        # l2
        metrics['depth_w/l2'] = F.mse_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w)
        # SSIM
        metrics['depth_w/ssim'] = K.metrics.ssim(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 11).mean()
        # PSNR
        metrics['depth_w/psnr'] = K.metrics.psnr(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 1)

    # by UW 
        #UW RGB
        # l1
        metrics['rgb_uw_by_uw_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw)
        # l2
        metrics['rgb_uw_by_uw_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw)
        # SSIM
        metrics['rgb_uw_by_uw_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 11).mean()
        # PSNR
        metrics['rgb_uw_by_uw_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 1)
        # W RGB
        # l1
        metrics['rgb_w_by_uw_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_uw)
        # l2
        metrics['rgb_w_by_uw_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_uw)
        # SSIM
        metrics['rgb_w_by_uw_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_uw, 11).mean()
        # PSNR
        metrics['rgb_w_by_uw_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_uw, 1)
        # by W
        #UW RGB
        #by W
        # l1
        metrics['rgb_uw_by_w_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw)
        # l2
        metrics['rgb_uw_by_w_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw)
        # SSIM
        metrics['rgb_uw_by_w_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 11).mean()
        # PSNR
        metrics['rgb_uw_by_w_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 1)

        #W RGB
        #by W
        # l1
        metrics['rgb_w_by_w_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_w)
        # l2
        metrics['rgb_w_by_w_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_w)
        # SSIM
        metrics['rgb_w_by_w_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_w, 11).mean()
        # PSNR
        metrics['rgb_w_by_w_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_w, 1)
    
        return metrics

    def _compute_losses(self, depths_pred_uw, depths_pred_w, \
                        rgb_uw, rgb_w, monodepth_uw, monodepth_w, mask_uw):
        losses = dict()

        for index in range(5):
            depth_i_pred_uw = depths_pred_uw[index]
            depth_i_pred_w = depths_pred_w[index]

            rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, rgb_w)
            rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, rgb_uw)

            rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, rgb_w)
            rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, rgb_uw)

            if self.photo_loss =='l1':
                losses[f'UW_l1_warp_loss_by_UW_depth_{index}'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw)
                losses[f'UW_l1_warp_loss_by_W_depth_{index}'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw)
                losses[f'W_l1_warp_loss_by_UW_depth_{index}'] = F.l1_loss(rgb_w,rgb_w_pred_by_depth_i_uw)
                losses[f'W_l1_warp_loss_by_W_depth_{index}'] = F.l1_loss(rgb_w,rgb_w_pred_by_depth_i_w)
            elif self.photo_loss =='l2':
                losses[f'UW_l2_warp_loss_by_UW_depth_{index}'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw)
                losses[f'UW_l2_warp_loss_by_W_depth_{index}'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw)
                losses[f'W_l2_warp_loss_by_UW_depth_{index}'] = F.mse_loss(rgb_w,rgb_w_pred_by_depth_i_uw)
                losses[f'W_l2_warp_loss_by_W_depth_{index}'] = F.mse_loss(rgb_w,rgb_w_pred_by_depth_i_w)
            elif self.photo_loss == 'ssim':
                losses[f'UW_ssim_warp_loss_by_UW_depth_{index}'] = self._ssim_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw).mean()
                losses[f'UW_ssim_warp_loss_by_W_depth_{index}'] = self._ssim_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw).mean()
                losses[f'W_ssim_warp_loss_by_UW_depth_{index}'] = self._ssim_loss(rgb_w,rgb_w_pred_by_depth_i_uw).mean()
                losses[f'W_ssim_warp_loss_by_W_depth_{index}'] = self._ssim_loss(rgb_w,rgb_w_pred_by_depth_i_w).mean()
            elif self.photo_loss == 'ssim_l1':
                losses[f'UW_ssim_l1_warp_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_uw*mask_uw).mean()
                losses[f'UW_ssim_l1_warp_loss_by_W_depth_{index}'] = self._ssim_l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_depth_i_w*mask_uw).mean()
                losses[f'W_ssim_l1_warp_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(rgb_w,rgb_w_pred_by_depth_i_uw).mean()
                losses[f'W_ssim_l1_warp_loss_by_W_depth_{index}'] = self._ssim_l1_loss(rgb_w,rgb_w_pred_by_depth_i_w).mean()

            if self.smooth_loss:
                losses[f'UW_smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_uw, rgb_uw)
                losses[f'W_smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_w, rgb_w)

            if self.structure_loss:
                losses[f'UW_structure_ssim_loss_{index}'] = self._ssim_loss(self._normalize_depth(depth_i_pred_uw), self._normalize_depth(monodepth_uw)).mean()
                losses[f'W_structure_ssim_loss_{index}']  = self._ssim_loss(self._normalize_depth(depth_i_pred_w), self._normalize_depth(monodepth_w)).mean()

            if self.depth_consist_loss:
                detached_depth_i_pred_uw = depth_i_pred_uw.detach()
                depth_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(detached_depth_i_pred_uw*self.depth_scale, depth_i_pred_w)
                depth_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, depth_i_pred_w)
                depth_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, detached_depth_i_pred_uw)
                depth_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(detached_depth_i_pred_uw*self.depth_scale, detached_depth_i_pred_uw)


                if self.depth_consist_loss =='l1':
                    losses[f'UW_l1_depth_consist_loss_by_UW_depth_{index}'] = F.l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_l1_depth_consist_loss_by_W_depth_{index}']  = F.l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_l1_depth_consist_loss_by_UW_depth_{index}']  = F.l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_l1_depth_consist_loss_by_W_depth_{index}']   = F.l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)
                elif self.depth_consist_loss =='l2':
                    losses[f'UW_l2_depth_consist_loss_by_UW_depth_{index}'] = F.mse_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_l2_depth_consist_loss_by_W_depth_{index}']  = F.mse_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_l2_depth_consist_loss_by_UW_depth_{index}']  = F.mse_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_l2_depth_consist_loss_by_W_depth_{index}']   = F.mse_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)
                elif self.depth_consist_loss == 'ssim':
                    losses[f'UW_ssim_depth_consist_loss_by_UW_depth_{index}'] = self._ssim_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_ssim_depth_consist_loss_by_W_depth_{index}']  = self._ssim_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_ssim_depth_consist_loss_by_UW_depth_{index}']  = self._ssim_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_ssim_depth_consist_loss_by_W_depth_{index}'  ] = self._ssim_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)
                elif self.depth_consist_loss == 'ssim_l1':
                    losses[f'UW_ssim_l1_depth_consist_loss_by_UW_depth_{index}'] = self._ssim_l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_uw*mask_uw)
                    losses[f'UW_ssim_l1_depth_consist_loss_by_W_depth_{index}']  = self._ssim_l1_loss(detached_depth_i_pred_uw*mask_uw, depth_uw_pred_by_depth_i_w*mask_uw)
                    losses[f'W_ssim_l1_depth_consist_loss_by_UW_depth_{index}']  = self._ssim_l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_uw)
                    losses[f'W_ssim_l1_depth_consist_loss_by_W_depth_{index}']   = self._ssim_l1_loss(depth_i_pred_w,          depth_w_pred_by_depth_i_w)

        return losses

    def _warp_train_data(self, gt_depth_uw, rgb_uw, ndepth_uw):

        tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw)
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        else:
            mask_uw = torch.ones_like(gt_depth_uw)
        
        model_input = {'x_img':rgb_uw,
                       'y_img': rgb_w}

        train_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_w = 1 - monodepth_w

            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth.shape[2:],
                                                mode="bicubic")
                monodepth_w = F.interpolate(monodepth_w, size=gt_depth.shape[2:],
                                                mode="bicubic")

            train_loss_data['monodepth_uw'] = monodepth_uw
            train_loss_data['monodepth_w'] = monodepth_w

        return model_input, train_loss_data

    def _epoch_fit(self, ):
        epoch_pbar = tqdm(iter(self.train_dataset))
        for gt_depth_uw, rgb_uw, ndepth_uw, conf in epoch_pbar:

            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()
            ndepth_uw = ndepth_uw.cuda()
            
            with torch.no_grad():
                model_input, train_loss_data = \
                    self._warp_train_data(gt_depth_uw, rgb_uw, ndepth_uw)
                del gt_depth_uw, ndepth_uw

            depths_pred_uw, depths_pred_w = self.model(**model_input)

            train_loss_data['depths_pred_uw'] = depths_pred_uw
            train_loss_data['depths_pred_w'] = depths_pred_w

            losses = self._compute_losses(**train_loss_data)

            total_loss = 0
            for loss_name in losses.keys():
                loss = losses[loss_name]
                weight = self.weights[loss_name]
                self.tb_writer.add_scalar('train_loss/{}'.format(loss_name), loss, self.n_iter)
                weighted_loss = loss * weight
                self.tb_writer.add_scalar('weighted_train_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
                total_loss += weighted_loss

            self.opt.zero_grad()
            if self.use_lr_decay:
                self._adjust_learning_rate()
            total_loss.backward()
            self.opt.step()

            self.tb_writer.add_scalar('train_loss/total_loss', total_loss, self.n_iter)

            self.n_iter+=1


    def _warp_val_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)

        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw)
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        else:
            mask_uw = torch.ones_like(gt_depth_uw)
        
        model_input = {'x_img':rgb_uw, 'y_img': rgb_w}

        val_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        val_metric_data = {'rgb_uw':rgb_uw,
                            'rgb_w': rgb_w,
                            'gt_depth_uw': gt_depth_uw,
                            'gt_depth_w': gt_depth_w,
                            'mask_uw': mask_uw,
                            'conf_uw': conf_uw,
                            'conf_w' : conf_w}

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_w = 1 - monodepth_w

            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
                monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")

            val_loss_data['monodepth_uw'] = monodepth_uw
            val_loss_data['monodepth_w'] = monodepth_w

        return model_input, val_loss_data, val_metric_data


    def _epoch_val(self,):
        print('epoch testing...')
        val_pbar = tqdm(iter(self.val_dataset))
        i=0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in val_pbar:
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()
            ndepth_uw = ndepth_uw.cuda()
            conf_uw = conf_uw.cuda()

            model_input, val_loss_data, val_metric_data = \
                self._warp_val_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            depths_pred_uw, depths_pred_w = self.model(**model_input)

            val_loss_data['depths_pred_uw'] = depths_pred_uw
            val_loss_data['depths_pred_w'] = depths_pred_w

            val_metric_data['depth_pred_uw'] = depths_pred_uw[-1]
            val_metric_data['depth_pred_w'] = depths_pred_w[-1]

            if i==0:
                total_losses = self._compute_losses(**val_loss_data)
    
                total_metrics = self._compute_val_metrics(**val_metric_data)

            else:
                losses = self._compute_losses(**val_loss_data)
                metrics = self._compute_val_metrics(**val_metric_data)

                for loss_name in losses.keys():
                    total_losses[loss_name] += losses[loss_name]

                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
            i+=1

        total_loss = 0
        weighted_total_loss = 0
        for loss_name in total_losses.keys():
            loss = total_losses[loss_name]/i
            self.tb_writer.add_scalar('val_loss/{}'.format(loss_name), loss, self.n_iter)
            total_loss+=loss 
            weighted_loss = loss * self.weights[loss_name]
            self.tb_writer.add_scalar('weighted_val_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
            weighted_total_loss += weighted_loss

        self.tb_writer.add_scalar('val_loss/total_loss', total_loss, self.n_iter)
        self.tb_writer.add_scalar('val_loss/weighted_total_loss', weighted_total_loss, self.n_iter)

        for metric_name in metrics.keys():
            metric = total_metrics[metric_name]/i
            self.tb_writer.add_scalar('metrics/{}'.format(metric_name), metric, self.n_iter)

            if metric_name == 'depth_uw/mae':
                if metric < self.uw_best_mae:
                    self.uw_best_mae_epoch = True
                    self.uw_best_mae = metric
                else:
                    self.uw_best_mae_epoch = False 

            if metric_name == 'depth_w/mae':
                if metric < self.w_best_mae:
                    self.w_best_mae_epoch = True
                    self.w_best_mae = metric
                else:
                    self.w_best_mae_epoch = False     

    def _epoch(self,):
        if self.fixed_img_encoder:
            self.model.E_img.train(False)
            self.model.D.train(True)
        else:
            self.model.train(True)
            self.model.train(True)
        self._epoch_fit()
        self.model.train(False)
        self.model.train(False)
        with torch.no_grad():
            self._visualize()
            self._epoch_val()
            if self.uw_best_mae_epoch or self.w_best_mae_epoch:
                self._save_model_cpt()
        self.epoch+=1

    def _visualize(self,):
        
        depths_pred_uw, depths_pred_w \
                = self.model(self.val_rgb_uw.cuda(),self.val_rgb_w.cuda())

        save_path = os.path.join(self.img_dir, '{}'.format(self.n_iter))
        os.makedirs(save_path, exist_ok=True)

        for index in trange(5):
            depth_i_pred_uw = depths_pred_uw[index]
            depth_i_pred_w = depths_pred_w[index]

            rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_w.cuda())
            rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_w.cuda())
            rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_uw.cuda())
            rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_uw.cuda())

            self.tb_writer.add_images(f'test/depth_uw_{index}', depth_i_pred_uw, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw, os.path.join(save_path, f'depth_pred_uw_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_uw', depth_i_pred_uw, save_path) 

            depth_i_pred_uw_conf = depth_i_pred_uw * self.val_conf_uw.cuda()
            self.tb_writer.add_images(f'test/depth_uw_{index}_conf', depth_i_pred_uw_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw_conf, os.path.join(save_path, f'depth_pred_uw_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_uw_conf', depth_i_pred_uw_conf, save_path) 

            self.tb_writer.add_images(f'test/depth_w_{index}', depth_i_pred_w, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_w, os.path.join(save_path, f'depth_pred_w_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_w', depth_i_pred_w, save_path) 
            
            depth_i_pred_w_conf = depth_i_pred_w * self.val_conf_w.cuda()
            self.tb_writer.add_images(f'test/depth_w_{index}_conf', depth_i_pred_w_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_w_conf, os.path.join(save_path, f'depth_pred_w_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_w_conf', depth_i_pred_w_conf, save_path) 

            self.tb_writer.add_images(f'val_rgb/rgb_uw_pred_by_depth_{index}_uw', rgb_uw_pred_by_depth_i_uw, self.n_iter)
            torchvision.utils.save_image(rgb_uw_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_uw.png'))  
            
            self.tb_writer.add_images(f'val_rgb/uw_pred_by_depth_{index}_w', rgb_uw_pred_by_depth_i_w, self.n_iter)
            torchvision.utils.save_image(rgb_uw_pred_by_depth_i_w, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_w.png'))  

            self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_w', rgb_w_pred_by_depth_i_w, self.n_iter)
            torchvision.utils.save_image(rgb_w_pred_by_depth_i_w, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_w.png'))  
            
            self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_uw', rgb_w_pred_by_depth_i_uw, self.n_iter)
            torchvision.utils.save_image(rgb_w_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_uw.png')) 


    def _save_model_cpt(self,):
        filename = 'E{}_iter_{}.cpt'.format(self.epoch, self.n_iter)
        if self.uw_best_mae_epoch or self.w_best_mae_epoch:
            filename = f'UW_W_best_{filename}'
        elif self.uw_best_mse_epoch:
            filename = f'UW_best_{filename}'
        elif self.w_best_mse_epoch:
            filename = f'W_best_{filename}'

        save_path = os.path.join(self.cpt_dir, filename)
        print('saving model cpt @ {}'.format(save_path))
        torch.save(self.model.state_dict(), save_path)


class TFT3D_TwoSplitStereoModelTrainer(TFT3D_TwoSplitModelTrainer):
    def __init__(self, uw_model, w_model):
        super().__init__(uw_model, w_model)
        self.uw_model  = uw_model.cuda() 
        self.w_model  = w_model.cuda() 

        self.n_iter = 0
        self.epoch = 0

    def _warp_train_data(self, gt_depth_uw, rgb_uw, ndepth_uw):
            
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw)
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        else:
            mask_uw = torch.ones_like(gt_depth_uw)
        
        uw_model_input = {'x_img':rgb_uw, 'y_img': rgb_w}

        w_model_input = {'x_img': rgb_w, 'y_img': rgb_uw}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            uw_model_input['x_depth']= depth_uw
            w_model_input['x_depth']= depth_w

        train_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_w = 1 - monodepth_w

            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
                monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")

            train_loss_data['monodepth_uw'] = monodepth_uw
            train_loss_data['monodepth_w'] = monodepth_w

        return uw_model_input, w_model_input, train_loss_data

    def _warp_val_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)

        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw)
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        else:
            mask_uw = torch.ones_like(gt_depth_uw)
        
        uw_model_input = {'x_img':rgb_uw, 'y_img': rgb_w}
        w_model_input = {'x_img': rgb_w, 'y_img': rgb_uw}

        val_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        val_metric_data = {'rgb_uw':rgb_uw,
                            'rgb_w': rgb_w,
                            'gt_depth_uw': gt_depth_uw,
                            'gt_depth_w': gt_depth_w,
                            'mask_uw': mask_uw,
                            'conf_uw': conf_uw,
                            'conf_w' : conf_w,
                            'tof_uw_mask': None,
                            'tof_w_mask': None }

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            uw_model_input['x_depth']= depth_uw
            w_model_input['x_depth']= depth_w
            tof_uw_mask = depth_uw==0
            tof_w_mask = depth_w==0
            val_metric_data['tof_uw_mask'] = tof_uw_mask
            val_metric_data['tof_w_mask'] = tof_w_mask

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_w = 1 - monodepth_w

            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
                monodepth_w = F.interpolate(monodepth_w, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")

            val_loss_data['monodepth_uw'] = monodepth_uw
            val_loss_data['monodepth_w'] = monodepth_w

        return uw_model_input, w_model_input, val_loss_data, val_metric_data 

    def _visualize(self,):
        
        if 'RGBD' in self.mode:
            depths_pred_uw = self.uw_model(self.val_rgb_uw.cuda(), 
                                           self.val_rgb_w.cuda(), 
                                           self.val_input_depth_uw.cuda())
            depths_pred_w = self.w_model(self.val_rgb_w.cuda(), 
                                         self.val_rgb_uw.cuda(), 
                                         self.val_input_depth_w.cuda())
        else:
            depths_pred_uw = self.uw_model(self.val_rgb_uw.cuda(), self.val_rgb_w.cuda())
            depths_pred_w = self.w_model(self.val_rgb_w.cuda(), self.val_rgb_uw.cuda())       

        save_path = os.path.join(self.img_dir, '{}'.format(self.n_iter))
        os.makedirs(save_path, exist_ok=True)

        for index in trange(5):
            depth_i_pred_uw = depths_pred_uw[index]
            depth_i_pred_w = depths_pred_w[index]
            
            rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_w.cuda())
            rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_w.cuda())
            rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_uw.cuda())
            rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_uw.cuda())

            self.tb_writer.add_images(f'test/depth_uw_{index}', depth_i_pred_uw, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw, os.path.join(save_path, f'depth_pred_uw_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_uw', depth_i_pred_uw, save_path) 

            depth_i_pred_uw_conf = depth_i_pred_uw * self.val_conf_uw.cuda()
            self.tb_writer.add_images(f'test/depth_uw_{index}_conf', depth_i_pred_uw_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw_conf, os.path.join(save_path, f'depth_pred_uw_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_uw_conf', depth_i_pred_uw_conf, save_path) 

            self.tb_writer.add_images(f'test/depth_w_{index}', depth_i_pred_w, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_w, os.path.join(save_path, f'depth_pred_w_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_w', depth_i_pred_w, save_path) 
            
            depth_i_pred_w_conf = depth_i_pred_w * self.val_conf_w.cuda()
            self.tb_writer.add_images(f'test/depth_w_{index}_conf', depth_i_pred_w_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_w_conf, os.path.join(save_path, f'depth_pred_w_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_w_conf', depth_i_pred_w_conf, save_path) 

            self.tb_writer.add_images(f'val_rgb/rgb_uw_pred_by_depth_{index}_uw', rgb_uw_pred_by_depth_i_uw, self.n_iter)
            torchvision.utils.save_image(rgb_uw_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_uw.png'))  
            
            self.tb_writer.add_images(f'val_rgb/uw_pred_by_depth_{index}_w', rgb_uw_pred_by_depth_i_w, self.n_iter)
            torchvision.utils.save_image(rgb_uw_pred_by_depth_i_w, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_w.png'))  

            self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_w', rgb_w_pred_by_depth_i_w, self.n_iter)
            torchvision.utils.save_image(rgb_w_pred_by_depth_i_w, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_w.png'))  
            
            self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_uw', rgb_w_pred_by_depth_i_uw, self.n_iter)
            torchvision.utils.save_image(rgb_w_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_uw.png')) 

class TFT3D_SingleSplitStereoModelTrainer(TFT3D_SingleSplitModelTrainer):
    def __init__(self, model, model_cam):
        super().__init__(model, model_cam)
        self.model  = model.cuda() 
        self.model_cam  = model_cam

        self.n_iter = 0
        self.epoch = 0


    def init_val_data(self,):
        gt_depth_uw, rgb_uw, ndepth_uw, conf_uw = next(iter(self.val_dataset))
        gt_depth_uw = gt_depth_uw.cuda()
        rgb_uw = rgb_uw.cuda()
        if 'RGBD' in self.mode:
            ndepth_uw = ndepth_uw.cuda()
        else:
            ndepth_uw = None
        conf_uw = conf_uw.cuda()

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)
        self.val_rgb_w = rgb_w.cpu()
        self.val_rgb_uw = rgb_uw.cpu()

        tof_depth = None
        self.val_conf_uw = None  
        self.val_conf_w = None
        self.val_input_depth_uw = None
        self.val_input_depth_w = None
        input_depth_uw = None
        input_depth_w = None
    
        if self.model_cam == 'w':
            gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
            conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)
            self.val_conf_w = conf_w.cpu()
        if self.model_cam == 'uw':
            self.val_conf_uw = conf_uw.cpu()          


        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            if self.model_cam == 'uw':
                input_depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                self.val_input_depth_uw = input_depth_uw.cpu()
            if self.model_cam == 'w':
                input_depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                self.val_input_depth_w = input_depth_w.cpu()


        save_path = os.path.join(self.img_dir, 'ground_truth')
        os.makedirs(save_path, exist_ok=True)

        if tof_depth is not None:
            tof_depth /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/tof_depth', tof_depth, self.n_iter)
            torchvision.utils.save_image(tof_depth, os.path.join(save_path, f'tof_depth.png')) 
            self._visualize_depths('ground_truth/tof_depth', tof_depth, save_path)


        if self.val_input_depth_uw is not None:
            self.tb_writer.add_images(f'input/depth_uw', self.val_input_depth_uw, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_uw, os.path.join(save_path, f'input_depth_uw.png')) 
            self._visualize_depths('input/depth_uw', self.val_input_depth_uw, save_path)
        if self.val_input_depth_w is not None:
            self.tb_writer.add_images(f'input/depth_w', self.val_input_depth_w, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_w, os.path.join(save_path, f'input_depth_w.png'))
            self._visualize_depths('input/depth_w', self.val_input_depth_w, save_path)

        if self.use_mask:
            mask_uw = torch.ones_like(gt_depth_uw) * self.depth_scale
            mask_uw = self.Wwarp2UW_by_UW_depth(mask_uw, mask_uw)>0
            mask_uw = mask_uw.float()
            self.tb_writer.add_images(f'input/mask_uw', mask_uw, self.n_iter)
            torchvision.utils.save_image(mask_uw, os.path.join(save_path, f'mask_uw.png'))

        self.tb_writer.add_images(f'ground_truth/rgb_uw', rgb_uw, self.n_iter)
        torchvision.utils.save_image(rgb_uw, os.path.join(save_path, f'rgb_uw.png')) 
        self.tb_writer.add_images(f'ground_truth/rgb_w', rgb_w, self.n_iter)
        torchvision.utils.save_image(rgb_w, os.path.join(save_path, f'rgb_w.png'))     

        if self.model_cam == 'uw':

            gt_depth_uw /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/gt_depth_uw', gt_depth_uw, self.n_iter)
            torchvision.utils.save_image(gt_depth_uw, os.path.join(save_path, f'gt_depth_uw.png')) 
            self._visualize_depths('gt_depth_uw/depth_uw', gt_depth_uw, save_path)

            gt_depth_uw_conf = gt_depth_uw * conf_uw
            self.tb_writer.add_images(f'ground_truth/gt_depth_uw_conf', gt_depth_uw_conf, self.n_iter)
            torchvision.utils.save_image(gt_depth_uw_conf, os.path.join(save_path, f'gt_depth_uw_conf.png')) 
            self._visualize_depths('gt_depth_uw/depth_uw_conf', gt_depth_uw_conf, save_path)

        if self.model_cam == 'w':

            gt_depth_w /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/gt_depth_w', gt_depth_w, self.n_iter)
            torchvision.utils.save_image(gt_depth_w, os.path.join(save_path, f'gt_depth_w.png')) 
            self._visualize_depths('gt_depth_w/depth_w', gt_depth_w, save_path)

            gt_depth_w_conf = gt_depth_w * conf_w
            self.tb_writer.add_images(f'ground_truth/gt_depth_w_conf', gt_depth_w_conf, self.n_iter)
            torchvision.utils.save_image(gt_depth_w_conf, os.path.join(save_path, f'gt_depth_w.png')) 
            self._visualize_depths('gt_depth_w/depth_w_conf', gt_depth_w, save_path)

    def _warp_train_data(self, gt_depth_uw, rgb_uw, ndepth_uw):
        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        mask_uw = torch.ones_like(gt_depth_uw)
        if self.use_mask:
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        
        if self.model_cam == 'uw':
            model_input = {'x_img':rgb_uw, 'y_img':rgb_w}

        if self.model_cam == 'w':
            model_input = {'x_img': rgb_w, 'y_img':rgb_uw}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            if self.model_cam == 'uw':
                depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth']= depth_uw
            if self.model_cam == 'w':
                depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth']= depth_w

        train_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        if self.structure_loss:
            if self.model_cam == 'uw':
                monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_uw = 1 - monodepth_uw
                if self.resize4midas:
                    monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth.shape[2:],
                                                    mode="bicubic")
                train_loss_data['monodepth_uw'] = monodepth_uw
            if self.model_cam == 'w':
                monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_w = 1 - monodepth_w
                if self.resize4midas:
                    monodepth_w = F.interpolate(monodepth_w, size=gt_depth.shape[2:],
                                                    mode="bicubic")
                train_loss_data['monodepth_w'] = monodepth_w

        return model_input, train_loss_data

    def _warp_val_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        mask_uw = torch.ones_like(gt_depth_uw)
        gt_depth_w = None
        conf_w = None

        if self.use_mask:
            mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        
        if self.model_cam == 'uw':
            model_input = {'x_img':rgb_uw, 'y_img':rgb_w}        

        if self.model_cam == 'w':
            gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
            conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)
            model_input = {'x_img': rgb_w, 'y_img':rgb_uw}

            
        val_loss_data = {'rgb_uw':rgb_uw,
                      'rgb_w': rgb_w,
                      'mask_uw': mask_uw,
                      'monodepth_uw': None,
                      'monodepth_w' : None}

        val_metric_data = {'rgb_uw':rgb_uw,
                            'rgb_w': rgb_w,
                            'gt_depth_uw': gt_depth_uw,
                            'gt_depth_w': gt_depth_w,
                            'mask_uw': mask_uw,
                            'conf_uw': conf_uw,
                            'conf_w' : conf_w}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            if self.model_cam == 'uw':
                depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth'] = depth_uw
                tof_uw_mask = depth_uw==0
                val_metric_data['tof_uw_mask'] = tof_uw_mask
            if self.model_cam == 'w':
                depth_w = self.Iwarp2W_by_I_depth(tof_depth, tof_depth)/self.depth_scale
                model_input['x_depth'] = depth_w
                tof_w_mask = depth_w==0
                val_metric_data['tof_w_mask'] = tof_w_mask

        if self.structure_loss:
            if self.model_cam == 'uw':
                monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
                monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_uw = 1 - monodepth_uw
                if self.resize4midas:
                    monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth.shape[2:],
                                                    mode="bicubic")
                val_loss_data['monodepth_uw'] = monodepth_uw
            if self.model_cam == 'w':
                monodepth_w = self.midas(self.midas_transform(rgb_w)).unsqueeze(1)
                monodepth_w = monodepth_w/monodepth_w.view(monodepth_w.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                monodepth_w = 1 - monodepth_w
                if self.resize4midas:
                    monodepth_w = F.interpolate(monodepth_w, size=gt_depth.shape[2:],
                                                    mode="bicubic")
                val_loss_data['monodepth_w'] = monodepth_w

        return model_input, val_loss_data, val_metric_data

    def _visualize(self,):
        
        if self.model_cam == 'uw':
            if self.val_input_depth_uw:
                depths_pred_uw = self.model(self.val_rgb_uw.cuda(), 
                                            self.val_rgb_w.cuda(),
                                            self.val_input_depth_uw.cuda())
            else:
                depths_pred_uw = self.model(self.val_rgb_uw.cuda(),
                                            self.val_rgb_w.cuda())

        if self.model_cam == 'w':
            if self.val_input_depth_w:
                depths_pred_w = self.model(self.val_rgb_w.cuda(), 
                                           self.val_rgb_uw.cuda(),
                                           self.val_input_depth_w.cuda())
            else:
                depths_pred_w = self.model(self.val_rgb_w.cuda(),
                                           self.val_rgb_uw.cuda())

        save_path = os.path.join(self.img_dir, '{}'.format(self.n_iter))
        os.makedirs(save_path, exist_ok=True)

        for index in trange(5):
            if self.model_cam == 'uw':
                depth_i_pred_uw = depths_pred_uw[index]
                rgb_uw_pred_by_depth_i_uw = self.Wwarp2UW_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_w.cuda())
                rgb_w_pred_by_depth_i_uw = self.UWwarp2W_by_UW_depth(depth_i_pred_uw*self.depth_scale, self.val_rgb_uw.cuda())
                
                self.tb_writer.add_images(f'test/depth_uw_{index}', depth_i_pred_uw, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_uw, os.path.join(save_path, f'depth_pred_uw_{index}.png'))
                self._visualize_depths(f'val_depth_{index}/depth_uw', depth_i_pred_uw, save_path) 
                
                depth_i_pred_uw_conf = depth_i_pred_uw * self.val_conf_uw.cuda()
                self.tb_writer.add_images(f'test/depth_uw_{index}_conf', depth_i_pred_uw_conf, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_uw_conf, os.path.join(save_path, f'depth_pred_uw_{index}_conf.png'))
                self._visualize_depths(f'val_depth_{index}/depth_uw_conf', depth_i_pred_uw_conf, save_path) 

                self.tb_writer.add_images(f'val_rgb/rgb_uw_pred_by_depth_{index}_uw', rgb_uw_pred_by_depth_i_uw, self.n_iter)
                torchvision.utils.save_image(rgb_uw_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_uw.png'))  
                
                self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_uw', rgb_w_pred_by_depth_i_uw, self.n_iter)
                torchvision.utils.save_image(rgb_w_pred_by_depth_i_uw, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_uw.png')) 

            if self.model_cam == 'w':
                depth_i_pred_w = depths_pred_w[index]
                rgb_uw_pred_by_depth_i_w = self.Wwarp2UW_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_w.cuda())
                rgb_w_pred_by_depth_i_w = self.UWwarp2W_by_W_depth(depth_i_pred_w*self.depth_scale, self.val_rgb_uw.cuda())

                self.tb_writer.add_images(f'test/depth_w_{index}', depth_i_pred_w, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_w, os.path.join(save_path, f'depth_pred_w_{index}.png'))
                self._visualize_depths(f'val_depth_{index}/depth_w', depth_i_pred_w, save_path) 
                
                depth_i_pred_w_conf = depth_i_pred_w * self.val_conf_w.cuda()
                self.tb_writer.add_images(f'test/depth_w_{index}_conf', depth_i_pred_w_conf, self.n_iter)
                torchvision.utils.save_image(depth_i_pred_w_conf, os.path.join(save_path, f'depth_pred_w_{index}_conf.png'))
                self._visualize_depths(f'val_depth_{index}/depth_w_conf', depth_i_pred_w_conf, save_path) 
                
                self.tb_writer.add_images(f'val_rgb/uw_pred_by_depth_{index}_w', rgb_uw_pred_by_depth_i_w, self.n_iter)
                torchvision.utils.save_image(rgb_uw_pred_by_depth_i_w, os.path.join(save_path, f'rgb_uw_pred_by_depth_{index}_w.png'))  

                self.tb_writer.add_images(f'val_rgb/w_pred_by_depth_{index}_w', rgb_w_pred_by_depth_i_w, self.n_iter)
                torchvision.utils.save_image(rgb_w_pred_by_depth_i_w, os.path.join(save_path, f'rgb_w_pred_by_depth_{index}_w.png'))  

        