import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.mesh import vert_normal


class VFNet(nn.Module):
    def __init__(self, C_in=1, C_hid=[16,32,32,32,32], M=2, R=3, K=3):
        super(VFNet, self).__init__()
        """
        A 3D U-Net to predit multiscale stationary velocity fields (SVFs).

        Args:
        - C_in: number of input channels
        - layers: number of hidden channels
        - M: number of SVFs for each resolution
        - R: number of scales
        - K: kernel size

        Inputs:
        - x: 3D brain MRI, (B,1,D1,D2,D3)

        Returns:
        - vf: multiscale SVFs, (M*R,3,D1,D2,D3)
        """
        assert R <= 3, 'number of scales should be <= 3'
        self.R = R
        self.M = M
        self.conv1 = nn.Conv3d(in_channels=C_in, out_channels=C_hid[0],
                               kernel_size=K, stride=1, padding=K//2)
        self.conv2 = nn.Conv3d(in_channels=C_hid[0], out_channels=C_hid[1],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv3 = nn.Conv3d(in_channels=C_hid[1], out_channels=C_hid[2],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv4 = nn.Conv3d(in_channels=C_hid[2], out_channels=C_hid[3],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv5 = nn.Conv3d(in_channels=C_hid[3], out_channels=C_hid[4],
                               kernel_size=K, stride=1, padding=K//2)

        self.deconv4 = nn.Conv3d(in_channels=C_hid[4]+C_hid[3], out_channels=C_hid[3],
                                 kernel_size=K, stride=1, padding=K//2)
        self.deconv3 = nn.Conv3d(in_channels=C_hid[3]+C_hid[2], out_channels=C_hid[2],
                                 kernel_size=K, stride=1, padding=K//2)
        self.deconv2 = nn.Conv3d(in_channels=C_hid[2]+C_hid[1], out_channels=C_hid[1],
                                 kernel_size=K, stride=1, padding=K//2)
        self.deconv1 = nn.Conv3d(in_channels=C_hid[1]+C_hid[0], out_channels=C_hid[0],
                                 kernel_size=K, stride=1, padding=K//2)

        self.flow1 = nn.Conv3d(in_channels=C_hid[2], out_channels=3*M,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow2 = nn.Conv3d(in_channels=C_hid[1], out_channels=3*M,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow3 = nn.Conv3d(in_channels=C_hid[0], out_channels=3*M,
                               kernel_size=K, stride=1, padding=K//2)
        
        nn.init.normal_(self.flow1.weight, 0, 1e-5)
        nn.init.constant_(self.flow1.bias, 0.0)
        nn.init.normal_(self.flow2.weight, 0, 1e-5)
        nn.init.constant_(self.flow2.bias, 0.0)
        nn.init.normal_(self.flow3.weight, 0, 1e-5)
        nn.init.constant_(self.flow3.bias, 0.0)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x):
        
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
                
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)

        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        vf1 = self.up(self.up(self.flow1(x)))
        # reshape to (M,3,D1,D2,D3)
        vf1 = vf1.reshape(self.M,3,*vf1.shape[2:])
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        vf2 = self.up(self.flow2(x))
        vf2 = vf2.reshape(self.M,3,*vf2.shape[2:])

        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        vf3 = self.flow3(x)
        vf3 = vf3.reshape(self.M,3,*vf3.shape[2:])

        if self.R == 3:
            vf = torch.cat([vf1, vf2, vf3], dim=0)
        elif self.R == 2:
            vf = torch.cat([vf2, vf3], dim=0)
        elif self.R == 1:
            vf = torch.cat([vf3], dim=0)
        return vf  # velocity field (M*R,3,D1,D2,D3)


class AttentionNet(nn.Module):
    """
    Channel-wise Attention Network.

    Args:
    - C_in: number of input channels
    - C: number of hidden channels
    - M: number of SVFs for each resolution
    - R: number of scales

    Inputs:
    - x: time sequence, (N,1)
    e.g., [0, 0.1, 0.2, ..., 1.0]
    
    Returns:
    - time-varying attention maps (N, M*R)
    """
    def __init__(self, C=16, M=2, R=3):
        super(AttentionNet, self).__init__()
        self.fc1 = nn.Linear(1,C*4)
        self.fc2 = nn.Linear(C*4,C*8)
        self.fc3 = nn.Linear(C*8,C*8)
        self.fc4 = nn.Linear(C*8,C*4)
        self.fc5 = nn.Linear(C*4,M*R)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = self.fc5(x)
        return F.softmax(x, dim=-1)  # (N, M*R)


class TANet(nn.Module):
    """
    Temporal Attention Network (TA-Net).

    Args:
    - C_in: number of input channels for VFNet
    - C_hid: number of hidden channels for VFNet
    - inshape: size of the input MRI volume
    - step_size: step size for integration
    - M: number of SVFs for each resolution
    - R: number of scales

    Inputs:
    - x: vertices of input mesh (1,|V|,3)
    - vol: 3D brain MRI, (1,1,D1,D2,D3)
    
    Returns:
    - x: vertices of deformed mesh (1,|V|,3)
    """
    def __init__(self,
                 C_in=1,
                 C_hid=[16,32,32,32,32],
                 inshape=[112,224,176],
                 step_size=0.02,
                 M=2,
                 R=3,
                 device='cuda:0'):
        
        super(TANet, self).__init__()
        # initialize neural network models
        self.vf_net = VFNet(C_in=C_in, C_hid=C_hid, M=M, R=R).to(device)
        self.att_net = AttentionNet(C=16, M=M, R=R).to(device)
        
        # image scale for grid sampling
        self.scale = torch.Tensor(inshape).to(device)[None,None,:] - 1  # (1,1,3)
        
        # for ODE integration
        self.h = step_size  # step size
        self.N = int(1/step_size)  # number of steps
        self.T = torch.arange(self.N)[:,None].to(device) * self.h  # time step
        
    def forward(self, x, vol):
        # ------ temporal attention ------ 
        # learn an attention map to weight SVFs
        weight = self.att_net(self.T)[...,None,None] # (N,M*R,1,1)
        
        # ------ stationary velocity fields ------
        svfs = self.vf_net(vol)  # (M*R,3,D1,D2,D3)
        
        # ------ integration ------
        for n in range(self.N):
            v = self.interpolate(x, svfs)  # sample velocity (M,|V|,3)
            v = (weight[n] * v).sum(0, keepdim=True)  # weighted by attention
            x = x + self.h * v  # deformation
        return x
    
    def interpolate(self, x, vol):
        coord = 2 * x / self.scale - 1  # rescale vertices to [-1,1]
        coord = coord.repeat(vol.shape[0],1,1)[:,:,None,None].flip(-1)  # (1,|V|,3) => (M*R,|V|,1,1,3)
        v = F.grid_sample(
            vol, coord, mode='bilinear',
            padding_mode='border', align_corners=True)  # velocity (M*R,3,|V|,1,1)
        return v[...,0,0].permute(0,2,1)  # (M*R, |V|,3)
    