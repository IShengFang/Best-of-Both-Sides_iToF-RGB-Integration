import os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia as K
from tqdm import tqdm

class BasicTester(nn.Module):

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

    def compute_depth_metrics(self, predict, ground_truth):
        '''
        borrow by https://github.com/dusty-nv/pytorch-depth/blob/master/metrics.py
        '''
        valid_mask = ground_truth>0
        predict = predict[valid_mask]
        ground_truth = ground_truth[valid_mask]

        abs_diff = (predict - ground_truth).abs()
        mse = torch.pow(abs_diff, 2).mean()
        rmse = torch.sqrt(mse).cpu().item()
        mae = abs_diff.mean().cpu().item()
        log_diff = torch.log10(predict) - torch.log10(ground_truth)
        lg10 = log_diff.abs().mean().cpu().item()
        rmse_log = torch.sqrt(torch.pow(log_diff, 2).mean()).cpu().item()
        absrel = float((abs_diff / ground_truth).mean())
        sqrel = float((torch.pow(abs_diff, 2) / ground_truth).mean())

        maxRatio = torch.max(predict / ground_truth, ground_truth / predict)
        delta1 = (maxRatio < 1.25).float().mean().cpu().item()
        delta2 = (maxRatio < 1.25 ** 2).float().mean().cpu().item()
        delta3 = (maxRatio < 1.25 ** 3).float().mean().cpu().item()
        return mse.cpu().item(), rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log

    def _compute_test_metrics(self, rgb_uw, rgb_w, 
                                    gt_depth_uw, gt_depth_w, mask_uw,
                                    conf_uw, conf_w, tof_uw_mask, tof_w_mask,
                                    depth_pred_uw=None, depth_pred_w=None,):

        metrics = dict()
        if depth_pred_uw is not None:
            rgb_uw_pred_by_uw = self.Wwarp2UW_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_w)
            rgb_w_pred_by_uw = self.UWwarp2W_by_UW_depth(depth_pred_uw*self.depth_scale, rgb_uw)
        if depth_pred_w is not None:
            rgb_uw_pred_by_w = self.Wwarp2UW_by_W_depth(depth_pred_w*self.depth_scale, rgb_w)
            rgb_w_pred_by_w = self.UWwarp2W_by_W_depth(depth_pred_w*self.depth_scale, rgb_uw)
        

        #metrics
        #UW depth
        if depth_pred_uw is not None:
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
            metrics['depth_uw/l1'] = F.l1_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale*conf_uw).cpu().item()
            # l2
            metrics['depth_uw/l2'] = F.mse_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale * conf_uw).cpu().item()
            # SSIM
            metrics['depth_uw/ssim'] = K.metrics.ssim(gt_depth_uw / self.depth_scale  * conf_uw, depth_pred_uw * conf_uw, 11).mean().cpu().item()
            # PSNR
            metrics['depth_uw/psnr'] = K.metrics.psnr(gt_depth_uw / self.depth_scale * conf_uw, depth_pred_uw * conf_uw, 1).cpu().item()

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
        if depth_pred_w is not None:
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
            metrics['depth_w/l1'] = F.l1_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w).cpu().item()
            # l2
            metrics['depth_w/l2'] = F.mse_loss(gt_depth_w * conf_w, depth_pred_w * self.depth_scale * conf_w).cpu().item()
            # SSIM
            metrics['depth_w/ssim'] = K.metrics.ssim(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 11).mean().cpu().item()
            # PSNR
            metrics['depth_w/psnr'] = K.metrics.psnr(gt_depth_w / self.depth_scale * conf_w, depth_pred_w * conf_w, 1).cpu().item()

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

        #RGB by UW
        if depth_pred_uw is not None:
            # l1
            metrics['rgb_uw_by_uw_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw).cpu().item()
            # l2
            metrics['rgb_uw_by_uw_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw).cpu().item()
            # SSIM
            metrics['rgb_uw_by_uw_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 11).mean().cpu().item()
            # PSNR
            metrics['rgb_uw_by_uw_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_uw*mask_uw, 1).cpu().item()

            #by UW
            # l1
            metrics['rgb_w_by_uw_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_uw).cpu().item()
            # l2
            metrics['rgb_w_by_uw_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_uw).cpu().item()
            # SSIM
            metrics['rgb_w_by_uw_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_uw, 11).mean().cpu().item()
            # PSNR
            metrics['rgb_w_by_uw_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_uw, 1).cpu().item()


        #RGB by W
        if depth_pred_w is not None:
            # l1
            metrics['rgb_uw_by_w_depth/l1'] = F.l1_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw).cpu().item()
            # l2
            metrics['rgb_uw_by_w_depth/l2'] = F.mse_loss(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw).cpu().item()
            # SSIM
            metrics['rgb_uw_by_w_depth/ssim'] = K.metrics.ssim(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 11).mean().cpu().item()
            # PSNR
            metrics['rgb_uw_by_w_depth/psnr'] = K.metrics.psnr(rgb_uw*mask_uw,rgb_uw_pred_by_w*mask_uw, 1).cpu().item()

            #W RGB
            #by W
            # l1
            metrics['rgb_w_by_w_depth/l1'] = F.l1_loss(rgb_w, rgb_w_pred_by_w).cpu().item()
            # l2
            metrics['rgb_w_by_w_depth/l2'] = F.mse_loss(rgb_w, rgb_w_pred_by_w).cpu().item()
            # SSIM
            metrics['rgb_w_by_w_depth/ssim'] = K.metrics.ssim(rgb_w, rgb_w_pred_by_w, 11).mean().cpu().item()
            # PSNR
            metrics['rgb_w_by_w_depth/psnr'] = K.metrics.psnr(rgb_w, rgb_w_pred_by_w, 1).cpu().item()
            
        return metrics

class BothModelTester(BasicTester):

    def __init__(self, cam_path, depth_scale, uw_model, w_model, mode):
        super().__init__()
        self.cam_path = cam_path
        self.load_camera_matrix(cam_path)
        self.depth_scale = depth_scale

        self.uw_model  = uw_model.cuda() 
        self.w_model  = w_model.cuda() 

        self.mode = mode

    def _warp_test_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)

        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)

        mask_uw = torch.ones_like(gt_depth_uw)
        mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0

        if self.mode == 'SplitStereoRGB':
            uw_model_input = {'x_img':rgb_uw, 'y_img': rgb_w}   
            w_model_input = {'x_img': rgb_w, 'y_img': rgb_uw}
        else:
            uw_model_input = {'x_img':rgb_uw}
            w_model_input = {'x_img': rgb_w}

        test_metric_data = {'rgb_uw':rgb_uw,
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
            test_metric_data['tof_uw_mask'] = tof_uw_mask
            test_metric_data['tof_w_mask'] = tof_w_mask


        return uw_model_input, w_model_input, test_metric_data
    
    @torch.no_grad()
    def test(self, test_dataset):
        print('dataset testing...')
        self.test_dataset = test_dataset
        test_pbar = tqdm(iter(self.test_dataset))
        i=0
        num_sample = 0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in test_pbar: 
            num_sample += gt_depth_uw.size(0)
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            conf_uw = conf_uw.cuda()

            uw_model_input, w_model_input, test_metric_data = \
                self._warp_test_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            depths_pred_uw = self.uw_model(**uw_model_input)
            depths_pred_w = self.w_model(**w_model_input)

            test_metric_data['depth_pred_uw'] = depths_pred_uw[-1]
            test_metric_data['depth_pred_w'] = depths_pred_w[-1]

            if i==0:
                total_metrics = self._compute_test_metrics(**test_metric_data)
                i+=1
            elif self.test_dataset.batch_size != gt_depth_uw.size(0):
                i_ratio = gt_depth_uw.size(0)/self.test_dataset.batch_size
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]*i_ratio
                i+=i_ratio
            else:
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
                i+=1

        for metric_name in metrics.keys():
            metrics[metric_name] = total_metrics[metric_name]/i

        return metrics
    
class StereoRGBTester(BasicTester):

    def __init__(self, cam_path, depth_scale, model):
        super().__init__()
        self.cam_path = cam_path
        self.load_camera_matrix(cam_path)
        self.depth_scale = depth_scale

        self.model  = model.cuda() 

    def _warp_test_data(self, gt_depth_uw, rgb_uw, conf_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)

        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)

        mask_uw = torch.ones_like(gt_depth_uw)
        mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        
        model_input = {'x_img':rgb_uw, 'y_img': rgb_w}

        test_metric_data = {'rgb_uw':rgb_uw,
                            'rgb_w': rgb_w,
                            'gt_depth_uw': gt_depth_uw,
                            'gt_depth_w': gt_depth_w,
                            'mask_uw': mask_uw,
                            'conf_uw': conf_uw,
                            'conf_w' : conf_w,
                            'tof_uw_mask': None,
                            'tof_w_mask': None}

        return model_input, test_metric_data

    @torch.no_grad()
    def test(self, test_dataset):
        print('dataset testing...')
        self.test_dataset = test_dataset
        test_pbar = tqdm(iter(self.test_dataset))
        i=0
        num_sample = 0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, _, conf_uw in test_pbar: 
            num_sample += gt_depth_uw.size(0)
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            conf_uw = conf_uw.cuda()

            model_input, test_metric_data = \
                self._warp_test_data(gt_depth_uw, rgb_uw, conf_uw)

            depths_pred_uw, depths_pred_w= self.model(**model_input)

            test_metric_data['depth_pred_uw'] = depths_pred_uw[-1]
            test_metric_data['depth_pred_w'] = depths_pred_w[-1]

            if i==0:
                total_metrics = self._compute_test_metrics(**test_metric_data)
                i+=1
            elif self.test_dataset.batch_size != gt_depth_uw.size(0):
                i_ratio = gt_depth_uw.size(0)/self.test_dataset.batch_size
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]*i_ratio
                i+=i_ratio
            else:
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
                i+=1

        for metric_name in metrics.keys():
            metrics[metric_name] = total_metrics[metric_name]/i

        return metrics


class SingleModelTester(BasicTester):

    def __init__(self, cam_path, depth_scale, model, test_cam, mode):
        super().__init__()
        self.cam_path = cam_path
        self.load_camera_matrix(cam_path)
        self.depth_scale = depth_scale

        self.model  = model.cuda() 

        self.mode = mode
        self.test_cam = test_cam 

    def _warp_test_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):

        rgb_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, rgb_uw)

        gt_depth_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, gt_depth_uw)
        conf_w = self.UWwarp2W_by_UW_depth(gt_depth_uw, conf_uw)

        mask_uw = torch.ones_like(gt_depth_uw)
        mask_uw = self.Wwarp2UW_by_UW_depth(gt_depth_uw, mask_uw)>0
        
        if self.mode == 'SplitStereoRGB':
            if self.test_cam == 'uw':
                model_input = {'x_img':rgb_uw, 'y_img': rgb_w}
            if self.test_cam == 'w':
                model_input = {'x_img': rgb_w, 'y_img': rgb_uw}
        else:
            if self.test_cam == 'uw':
                model_input = {'x_img':rgb_uw}
            if self.test_cam == 'w':
                model_input = {'x_img': rgb_w}

        test_metric_data = {'rgb_uw':rgb_uw,
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
            if self.test_cam == 'uw':
                model_input['x_depth']= depth_uw
            else:
                model_input['x_depth']= depth_w
            
            tof_uw_mask = depth_uw==0
            tof_w_mask = depth_w==0
            test_metric_data['tof_uw_mask'] = tof_uw_mask
            test_metric_data['tof_w_mask'] = tof_w_mask


        return model_input, test_metric_data
    
    @torch.no_grad()
    def test(self, test_dataset):
        print('dataset testing...')
        self.test_dataset = test_dataset
        test_pbar = tqdm(iter(self.test_dataset))
        i=0
        num_sample = 0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in test_pbar: 
            num_sample += gt_depth_uw.size(0)
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            conf_uw = conf_uw.cuda()

            model_input, test_metric_data = \
                self._warp_test_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            if self.test_cam == 'uw':
                depths_pred_uw = self.model(**model_input)
                test_metric_data['depth_pred_uw'] = depths_pred_uw[-1]
            if self.test_cam == 'w':
                depths_pred_w = self.model(**model_input)
                test_metric_data['depth_pred_w'] = depths_pred_w[-1]

            if i==0:
                total_metrics = self._compute_test_metrics(**test_metric_data)
                i+=1
            elif self.test_dataset.batch_size != gt_depth_uw.size(0):
                i_ratio = gt_depth_uw.size(0)/self.test_dataset.batch_size
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]*i_ratio
                i+=i_ratio
            else:
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
                i+=1

        for metric_name in metrics.keys():
            metrics[metric_name] = total_metrics[metric_name]/i

        return metrics