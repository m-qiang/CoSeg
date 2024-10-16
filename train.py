import numpy as np
from tqdm import tqdm
import nibabel as nib
import glob
import trimesh
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import chamfer_distance
from skimage.measure import marching_cubes
from net.loss import boundary_loss
from net.tanet import TANet

from utils.mesh import (
    adjacent_faces,
    vert_normal,
    face_normal,
    taubin_smooth,
)


class SurfDataset(Dataset):
    """
    Dataset class for surface reconstruction
    """
    def __init__(self, args, data_split='train'):
        super(SurfDataset, self).__init__()
        
        # ------ load arguments ------ 
        data_type = args.data_type
        surf_hemi = args.surf_hemi
        surf_type = args.surf_type
        step_size = args.step_size
        M = args.n_svf
        R = args.n_res
        device = args.device
        tag = args.tag

        subj_list = sorted(glob.glob(
            './data/'+data_type+'/'+data_split+'/*'))

        # ------ load template input ------ 
        if data_type == 'hcp':
            mesh_init = trimesh.load(
                './template/hcp_hemi-'+surf_hemi+'_init_160k.obj', process=False)
        elif data_type == 'dhcp':
            mesh_init = trimesh.load(
                './template/dhcp_fetal_hemi-'+surf_hemi+'_init_135k.obj', process=False)

        vert_init = mesh_init.vertices
        face_init = mesh_init.faces
        if surf_hemi == 'left':
            vert_init[:,0] = vert_init[:,0] - 64
        
        # ------ load pre-trained model ------
        if surf_type == 'pial':
            tanet = TANet(
                C_in=1, C_hid=[16,32,64,128,128], inshape=[112,224,176],
                step_size=step_size, M=M, R=R, device=device)
            model_dir = './ckpts/'+data_type+'/model_'+data_type+\
                        '_hemi-'+surf_hemi+'_white_'+tag+'_200epochs.pt'
            tanet.load_state_dict(
                torch.load(model_dir, map_location=device))
        
        self.data_list = []
        
        for i in tqdm(range(len(subj_list[:]))):
            subj_dir = subj_list[i]
            subj_id = subj_list[i].split('/')[-1]

            # ------ load gt segmentation ------
            if data_type == 'hcp':
                seg_gt = nib.load(
                    subj_dir+'/'+subj_id+'.ribbon.nii.gz')
            elif data_type == 'dhcp':
                seg_gt = nib.load(
                    subj_dir+'/'+subj_id+'_ribbon_affine.nii.gz')
            seg_gt = seg_gt.get_fdata()

            # clip left/right hemisphere
            if surf_hemi == 'left':
                seg_gt = seg_gt[None, 64:]
                seg_wm = (seg_gt == 1).astype(np.float32)
                seg_gm = (seg_gt == 3).astype(np.float32)
            elif surf_hemi == 'right':
                seg_gt = seg_gt[None, :64]
                seg_wm = (seg_gt == 2).astype(np.float32)
                seg_gm = (seg_gt == 4).astype(np.float32)
                
            # ------ load input volume ------
            if data_type == 'hcp':
                mri_in = nib.load(
                    subj_dir+'/'+subj_id+'.T1w_restore.nii.gz')
            elif data_type == 'dhcp':
                mri_in = nib.load(
                    subj_dir+'/'+subj_id+'_T2w_affine.nii.gz')
            mri_in = mri_in.get_fdata()
            mri_in = mri_in / mri_in.max()

            if surf_hemi == 'left':
                mri_in = mri_in[None, 64:]
            elif surf_hemi == 'right':
                mri_in = mri_in[None, :64]
            vol_in = mri_in.astype(np.float32)

            # ------ load input surface ------
            if surf_type == 'white':
                vert_in = vert_init.copy()
                face_in = face_init.copy()
            elif surf_type == 'pial':
                # use predicted white surface as the input
                vert_in = vert_init.copy()
                face_in = face_init.copy()
                vert_in = torch.Tensor(vert_in[None]).to(device)
                face_in = torch.LongTensor(face_in[None]).to(device)
                vol_wm = mri_in.astype(np.float32)
                vol_wm = torch.Tensor(vol_wm[None]).float().to(device)
                with torch.no_grad():
                    vert_wm = tanet(vert_in, vol_wm)
                    vert_wm = taubin_smooth(vert_wm, face_in, n_iters=10)
                vert_in = vert_wm[0].cpu().numpy()
                face_in = face_in[0].cpu().numpy()

            # ------ extract gt surface ------
            if surf_type == 'white':
                seg_gt = seg_wm.copy()
            elif surf_type == 'pial':
                seg_gt = seg_wm + seg_gm
            vert_gt, face_gt, _, _ = marching_cubes(seg_gt[0], level=0.5)
            vert_gt = vert_gt.copy()
            face_gt = face_gt[:,[2,1,0]]
            vert_gt = torch.Tensor(vert_gt[None]).to(device)
            face_gt = torch.LongTensor(face_gt[None]).to(device)
            vert_gt = taubin_smooth(vert_gt, face_gt, n_iters=100)
            vert_gt = vert_gt[0].cpu().numpy()
            face_gt = face_gt[0].cpu().numpy()

            surf_data = (vol_in, vert_in, vert_gt, face_in, face_gt)
            self.data_list.append(surf_data)  # add to data list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        surf_data = self.data_list[i]
        return surf_data
    
    
def train_loop(args):
    # ------ load arguments ------ 
    data_type = args.data_type  # type of the dataset
    surf_type = args.surf_type  # white or pial
    surf_hemi = args.surf_hemi  # left or right
    tag = args.tag
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    w_nc = args.w_nc  # weight for nc loss
    w_edge = args.w_edge  # weight for edge loss
    w_inflate = args.w_inflate  # weight for inflate loss
    M = args.n_svf  # number of SVFs
    R = args.n_res  # number of resolution levels
    step_size = args.step_size  # step size for integration
    
    # start training logging
    logging.basicConfig(
        filename='./ckpts/'+data_type+'/log_'+data_type+\
        '_hemi-'+surf_hemi+'_'+surf_type+'_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = SurfDataset(args, data_split='train')
    validset = SurfDataset(args, data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # ------ pre-compute adjacency------
    if data_type == 'hcp':
        mesh_in = trimesh.load(
            './template/hcp_hemi-'+surf_hemi+'_init_160k.obj', process=False)
    elif data_type == 'dhcp':
        mesh_in = trimesh.load(
            './template/dhcp_fetal_hemi-'+surf_hemi+'_init_135k.obj', process=False)

    face_in = mesh_in.faces
    face_in = torch.LongTensor(face_in[None]).to(device)
    # for normal consistency loss
    adj_faces = adjacent_faces(face_in)
    # for edge length loss
    edge_in = torch.cat([face_in[0,:,[0,1]],
                         face_in[0,:,[1,2]],
                         face_in[0,:,[2,0]]], dim=0).T
    
    # ------ initialize model ------ 
    logging.info("initalize model ...")
    
    if surf_type == 'white':
        C_hid = [16,32,64,128,128]  # number of channels for each layer
    elif surf_type == 'pial':
        C_hid = [16,32,32,32,32]  # fewer params to avoid overfitting
    tanet = TANet(
        C_in=1, C_hid=C_hid, inshape=[112,224,176],
        step_size=step_size, M=M, R=R, device=device)
    optimizer = optim.Adam(tanet.parameters(), lr=lr)

    # ------ training loop ------ 
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            vol_in, vert_in, vert_gt, face_in, face_gt = data
            vol_in = vol_in.to(device).float()
            vert_in = vert_in.to(device).float()
            face_in = face_in.to(device).long()
            vert_gt = vert_gt.to(device).float()
            face_gt = face_gt.to(device).long()
            optimizer.zero_grad()
            vert_pred = tanet(vert_in, vol_in)

            # normal consistency loss
            normal = face_normal(vert_pred, face_in)  # face normal
            nc_loss = (1 - normal[:,adj_faces].prod(-2).sum(-1)).mean()
            # edge loss
            vert_i = vert_pred[:,edge_in[0]]
            vert_j = vert_pred[:,edge_in[1]]
            edge_loss = ((vert_i - vert_j)**2).sum(-1).mean() 
            # reconstruction loss
            if surf_type == 'white':
                recon_loss = chamfer_distance(vert_pred, vert_gt)[0]
            elif surf_type == 'pial':
                if epoch <= 20:
                    # pre-training to ensure the white surface inflates
                    vert_gt = vert_in.clone()
                    for j in range(10):  # 0.5mm inflation
                        vert_gt += 0.1*vert_normal(vert_gt, face_in)
                    recon_loss = nn.MSELoss()(vert_pred, vert_gt)
                else:
                    # make sure the surface deform along normal direction
                    normal_in = vert_normal(vert_in, face_in)
                    # displacement
                    disp = vert_pred - vert_in
                    disp = disp / (torch.norm(disp, dim=-1).unsqueeze(-1) + 1e-12)
                    # the displacement has the same direction as the normal vector
                    inflate_loss = (1 - (normal_in * disp).sum(-1)).mean()
                    recon_loss = boundary_loss(vert_pred, vert_gt)[0]
                    recon_loss += w_inflate * inflate_loss
                    
            loss = recon_loss + w_nc*nc_loss + w_edge*edge_loss

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 20 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                recon_error = []
                for idx, data in enumerate(validloader):
                    vol_in, vert_in, vert_gt, face_in, face_gt = data
                    vol_in = vol_in.to(device).float()
                    vert_in = vert_in.to(device).float()
                    face_in = face_in.to(device).long()
                    vert_gt = vert_gt.to(device).float()
                    face_gt = face_gt.to(device).long()
                    
                    vert_pred = tanet(vert_in, vol_in)
                    if surf_type == 'white':
                        recon_loss = chamfer_distance(vert_pred, vert_gt)[0]
                        recon_error.append(recon_loss.item())
                    elif surf_type == 'pial':
                        recon_loss = boundary_loss(vert_pred, vert_gt)[0]
                        recon_error.append(recon_loss.item())
                    
                # save input/gt/predicted meshes to examine the training performance
                mesh_save = trimesh.Trimesh(vert_in[0].cpu().numpy(),
                                            face_in[0].cpu().numpy(),
                                            process=False)
                mesh_save.export(
                    './ckpts/'+data_type+'/surface_'+data_type+\
                    '_hemi-'+surf_hemi+'_'+surf_type+'_in.obj');
                
                mesh_save = trimesh.Trimesh(vert_gt[0].cpu().numpy(),
                                            face_gt[0].cpu().numpy(),
                                            process=False)
                mesh_save.export(
                    './ckpts/'+data_type+'/surface_'+data_type+\
                    '_hemi-'+surf_hemi+'_'+surf_type+'_gt.obj');
                
                mesh_save = trimesh.Trimesh(vert_pred[0].cpu().numpy(),
                                            face_in[0].cpu().numpy(),
                                            process=False)
                mesh_save.export(
                    './ckpts/'+data_type+'/surface_'+data_type+\
                    '_hemi-'+surf_hemi+'_'+surf_type+'_pred_'+tag+'.obj');

            logging.info('epoch:{}'.format(epoch))
            logging.info('recon error:{}'.format(np.mean(recon_error)))
            logging.info('-------------------------------------')
        
            # save model checkpoints
            torch.save(
                tanet.state_dict(), 
                './ckpts/'+data_type+'/model_'+data_type+\
                '_hemi-'+surf_hemi+'_'+surf_type+'_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="TANet Training")
    
    parser.add_argument('--data_type', default='hcp', type=str, help="[dhcp, hcp]")
    parser.add_argument('--surf_type', default='white', type=str, help="[white, pial]")
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda", type=str, help="[cuda, cpu]")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")

    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=100, type=int, help="number of training epochs")

    parser.add_argument('--w_nc', default=2.5, type=float, help="weight for normal consistency loss")
    parser.add_argument('--w_edge', default=0.5, type=float, help="weight for edge length loss")
    parser.add_argument('--w_inflate', default=5.0, type=float, help="weight for inflation loss")
    parser.add_argument('--n_svf', default=2, type=int, help="number of stationary velocity fields")
    parser.add_argument('--n_res', default=3, type=int, help="number of resolution levels")
    parser.add_argument('--step_size', default=0.02, type=float, help="integration step size")
    
    args = parser.parse_args()
    
    train_loop(args)