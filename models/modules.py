import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

class AdaptiveInstanceNorm(nn.Module):
    '''
    Module Interface for AdaIN(Adaptive Instance Normalization)    
    '''
    def calc_mean_std(self, x, eps=1e-5):
        x = x.reshape(x.size(0), -1)
        return x.mean(1, keepdim=True), x.std(1, keepdim=True)


    def adaptive_instance_norm(self, feat, adaptor, eps=1e-5):
        size = feat.size()
        feat_mean, feat_std = self.calc_mean_std(feat)
        adaptive_mean, adaptive_std = self.calc_mean_std(adaptor)

        normalized_feat = (feat - feat_mean) / feat_std
        adapted_feat = (normalized_feat*adaptive_std)+adaptive_mean
        return adapted_feat
    
    def __init__(self, eps=1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        '''
        x: feature for normalization
        y: feature for adaption
        '''
        return self.adaptive_instance_norm(x,y, eps=self.eps)
    
class ConditionalBatchNorm2d(nn.Module):
    '''
    code credit: https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    '''
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = y.chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class Conv2dModule(nn.Module):
    def __init__(self, in_dim , out_dim, k, s, p, norm='none', act='relu', leak=0.2, sn=False, transpose=False):
        '''
        norm : bn in |ln cbn adain adalin cln
        act: relu lrelu selu, elu
        '''
        super().__init__()
        self.transpose = transpose
        if self.transpose:
            self.conv = nn.ConvTranspose2d(in_dim, out_dim, k, s, p)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, k, s, p)

        if sn:
            self.conv = spectral_norm(self.conv)

        self.conditional = False

        # normalization
        if norm =='none':
            self.norm = None
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm == 'ln':
            raise NotDebugYet
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.conditional = True

        # conditional normalization
        if norm == 'cbn':
            self.norm = ConditionalBatchNorm2d(out_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm()
        elif norm == 'adalin':
            raise NotImplementYet
        elif norm == 'cln':
            raise NotImplementYet

        if act=='none':
            self.act = None
        elif act=='relu':
            self.act = nn.ReLU()
        elif act=='lrelu':
            self.act = nn.LeakyReLU(leak)
        elif act=='selu':
            self.act = nn.SELU()
        elif act=='elu':
            self.act = nn.ELU()
        elif act=='tanh':
            self.act = nn.Tanh()
        elif act=='sigmoid':
            self.act = nn.Sigmoid()
        elif act=='softplus':
            self.act = nn.Softplus()
        elif act=='relu6':
            self.act = nn.ReLU6()

    def forward(self,x, c=None):
        h = self.conv(x)
        if self.norm:
            if self.conditional:
                h = self.norm(h, c)
            else:
                h = self.norm(h)

        if self.act:
            h = self.act(h)

        return h

class Resblock(nn.Module):
    def __init__(self, dim, norm='none', act='relu', sn=False):
        super().__init__()
        
        self.conv_0 = Conv2dModule(dim, dim, 3,1,1, norm=norm, act=act, sn=sn)
        self.conv_1 = Conv2dModule(dim, dim, 3,1,1, norm=norm, act='none', sn=sn)

    def forward(self, x, c=None):
        h = self.conv_0(x,c)
        h = self.conv_1(h,c)

        return x+h


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
# modified form https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
class SAGAN_Attention(nn.Module):
    def __init__(self, ch, sn=False, name='attention'):
        super(Attention, self).__init__()
        self.sn = sn
        # Channel multiplier
        self.ch = ch
        #queary
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        #key
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        #value
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)

        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

        if sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        
    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

    def beta_forward(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return beta

    def o_forward(self,x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return o


class Self_Attention(nn.Module):
    def __init__(self, input_dim, k_reduce=8, q_reduce=8, v_reduce=1, sn=False):
        super().__init__()
        self.sn = sn
        # Channel multiplier
        self.input_dim = input_dim
        self.q_reduce = q_reduce
        self.k_reduce = k_reduce
        self.v_reduce = v_reduce
        #queary
        self.Q = nn.Conv2d(self.input_dim, self.input_dim // self.q_reduce, kernel_size=1, padding=0, bias=False)
        #key
        self.K = nn.Conv2d(self.input_dim, self.input_dim // self.k_reduce, kernel_size=1, padding=0, bias=False)
        #value
        self.V = nn.Conv2d(self.input_dim, self.input_dim // self.v_reduce, kernel_size=1, padding=0, bias=False)

        self.O = nn.Conv2d(self.input_dim // self.v_reduce, self.input_dim, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

        if sn:
            self.Q = spectral_norm(self.Q)
            self.K = spectral_norm(self.K)
            self.V = spectral_norm(self.V)
            self.O = spectral_norm(self.O)

    def e_forward(self, x):
        '''
        energy forward
        '''
        N, C, H, W = x.shape

        C_q = C//self.q_reduce
        C_k = C//self.k_reduce
        C_v = C//self.v_reduce
        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        #reshape HW
        q = q.view(N, C_q, H*W)
        k = k.view(N, C_k, H*W)
        v = v.view(N, C_v, H*W)
        #compute energy and attention
        e = torch.bmm(q.transpose(1, 2), k) #e shape is (N, H*W, H*W)

        return e

    def a_forward(self, x):
        '''
        attention forward
        '''
        return F.softmax(self.e_forward(x), dim=-1)

    def o_forward(self, x):
        '''
        forward w/o gamma
        '''
        N, C, H, W = x.shape

        C_q = C//self.q_reduce
        C_k = C//self.k_reduce
        C_v = C//self.v_reduce
        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        #reshape HW
        q = q.view(N, C_q, H*W)
        k = k.view(N, C_k, H*W)
        v = v.view(N, C_v, H*W)
        #compute energy and attention
        e = torch.bmm(q.transpose(1, 2), k) #e shape is (N, H*W, H*W)
        a = F.softmax(e, dim=-1)

        o = torch.bmm(v,a) #o shape (N, C_v, H*W)
        o = o.view(N, -1 , W, H)#o shape (N, C_v, H*W)
        o = self.O(o)

        return o

    def forward(self, x):
        return self.gamma*self.o_forward(x)


class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
                                
        
        # bilinear interpolate options
        self._up_kwargs = up_kwargs
        
        ## peter config:
        self.mode = 2
        self.PPM_ATTENTION = True
        self.CAM_ATTENTION = True
        self.CAM_ATTENTION_SCALE = True
        
        if self.mode == 0:
            self.fc_attention = FA_Module(in_dim=inter_channels)
        elif self.mode == 1 or self.mode == 2:
            self.fc_full_attention = FA_Full_Module(in_dim=inter_channels)
        else:
            raise (RuntimeError("mode should be 0 or 1"))
            
        if self.PPM_ATTENTION == True:
            self.ppm_attention1 = PAM_Module(in_dim=inter_channels)
            self.ppm_attention2 = PAM_Module(in_dim=inter_channels)
            self.ppm_attention3 = PAM_Module(in_dim=inter_channels)
        
        if self.CAM_ATTENTION == True:
            self.cam_attention = CAM_Module(in_dim=in_channels)
        
        if self.CAM_ATTENTION == True:
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels + inter_channels*2, in_channels, 1, bias=False),
                                    norm_layer(in_channels))
        else:
            self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                    norm_layer(in_channels))
        
        if self.CAM_ATTENTION_SCALE == True:
            self.cam_attention_scale = CAM_Module(in_dim=inter_channels*3)
            self.conv_cross = nn.Sequential(nn.Conv2d(inter_channels*3, inter_channels, 1, bias=False),
                                    norm_layer(inter_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        
        
        ### peter extension for strip pooling attention
        
        if self.mode == 0: # peter -> low gpu memory : attention on pooling layer
            
            ## old version #######
            #fc_h = self.pool3(x2) 
            #fc_w = self.pool4(x2) 
            
            #fc_h_attention = self.fc_attention(fc_h)
            #fc_w_attention = self.fc_attention(fc_w)
            
            #x2_4 = F.interpolate(self.conv2_3(fc_h_attention), (h, w), **self._up_kwargs) # peter
            #x2_5 = F.interpolate(self.conv2_4(fc_w_attention), (h, w), **self._up_kwargs) # peter
            ## old version #######
            
            fc_h = self.conv2_3(self.pool3(x2))
            fc_w = self.conv2_4(self.pool4(x2))
            
            fc_h_attention = self.fc_attention(fc_h)
            fc_w_attention = self.fc_attention(fc_w)
            
            x2_4 = F.interpolate(fc_h_attention, (h, w), **self._up_kwargs) # peter
            x2_5 = F.interpolate(fc_w_attention, (h, w), **self._up_kwargs) # peter
            
        
        elif self.mode == 1: # jerry -> high gpu memory : attention on input feature maps (attention on the height dimenstion and then pooling on the "height" dimension)
            
            x2_height_attention = self.fc_full_attention(x2, attention_with = "height")
            x2_width_attention = self.fc_full_attention(x2, attention_with = "width")
            
            x2_4 = F.interpolate(self.conv2_3(self.pool3(x2_height_attention)), (h, w), **self._up_kwargs)
            x2_5 = F.interpolate(self.conv2_4(self.pool4(x2_width_attention)), (h, w), **self._up_kwargs)
            
        elif self.mode == 2: # jerry -> high gpu memory : attention on input feature maps (attention on the height dimenstion and then pooling on the "width" dimension)
            
            x2_height_attention = self.fc_full_attention(x2, attention_with = "height")
            x2_width_attention = self.fc_full_attention(x2, attention_with = "width")
            
            x2_4 = F.interpolate(self.conv2_3(self.pool3(x2_width_attention)), (h, w), **self._up_kwargs)
            x2_5 = F.interpolate(self.conv2_4(self.pool4(x2_height_attention)), (h, w), **self._up_kwargs)
        
        else:
            raise (RuntimeError("mode should be 0 or 1"))
        
        ### end peter extension for strip pooling attention
        
        
        ### original
        #x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        #x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        ### end original
        
        
        ### peter: PPM_ATTENTION 
        if self.PPM_ATTENTION == True:
            
            x2_1 = self.ppm_attention1(x2_1)
            x2_2 = self.ppm_attention2(x2_2)
            x2_3 = self.ppm_attention3(x2_3)

        ### end peter: PPM_ATTENTION 
        
        if self.CAM_ATTENTION == True:
            #print(x.shape)
            x_attention = self.cam_attention(x)
        
        if self.CAM_ATTENTION_SCALE == True:
            
            x2_123 = torch.cat([x2_1, x2_2, x2_3], dim=1)
            x2_123 = self.cam_attention_scale(x2_123)

            x1 = self.conv2_5(F.relu_(self.conv_cross(x2_123)))
            
        else:
            x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        
        
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4)) # no softmax??
        
        if self.CAM_ATTENTION == True:
            out = self.conv3(torch.cat([x_attention, x1, x2], dim=1))
        else:
            out = self.conv3(torch.cat([x1, x2], dim=1))
        
        return F.relu_(x + out)
