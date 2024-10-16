import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import nibabel as nib
import argparse
import time
import trimesh
from net.tanet import TANet
from utils.mesh import apply_affine_mat, taubin_smooth
from utils.io import save_gifti_surface



if __name__ == "__main__":
    
    # ------ load arguments ------ 
    parser = argparse.ArgumentParser(description="CoSeg")
    
    parser.add_argument('--data_dir', default='./dataset/', type=str, help="directory of the input")
    parser.add_argument('--model_dir', default='./model/', type=str, help="directory of the saved models")
    parser.add_argument('--save_dir', default='./result/', type=str, help="directory to save the surfaces")
    parser.add_argument('--data_type', default='dhcp', type=str, help="[dhcp, hcp]")
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda:0", type=str, help="cuda or cpu")
    parser.add_argument('--n_svf', default=2, type=int, help="number of stationary velocity fields")
    parser.add_argument('--n_res', default=3, type=int, help="number of resolution levels")
    parser.add_argument('--step_size', default=0.02, type=float, help="integration step size")

    args = parser.parse_args()
    
    data_dir = args.data_dir  # directory of the input mri
    model_dir = args.model_dir  # directory of the saved models
    save_dir = args.save_dir  # directory to save the surface
    data_type = args.data_type  # dhcp or hcp
    surf_hemi = args.surf_hemi
    device = torch.device(args.device)
    step_size = args.step_size
    M = args.n_svf
    R = args.n_res
    
    # ------- load template ------- 
    print('Load template surface ...')
    if data_type == 'hcp':
        mesh_in = trimesh.load(
            './template/hcp_hemi-'+surf_hemi+'_init_160k.obj', process=False)
    elif data_type == 'dhcp':
        mesh_in = trimesh.load(
            './template/dhcp_fetal_hemi-'+surf_hemi+'_init_135k.obj', process=False)
    vert_in = mesh_in.vertices
    face_in = mesh_in.faces
    if surf_hemi == 'left':
        vert_in[:,0] = vert_in[:,0] - 64
    vert_in = torch.Tensor(vert_in[None]).to(device)
    face_in = torch.LongTensor(face_in[None]).to(device)
    
    # ------ load input volume ------
    mri_in = nib.load(data_dir)
    affine_in = mri_in.affine
    mri_in = mri_in.get_fdata()
    mri_in = mri_in / mri_in.max()

    if surf_hemi == 'left':
        mri_in = mri_in[None, 64:]
    elif surf_hemi == 'right':
        mri_in = mri_in[None, :64]
    vol_in = mri_in.astype(np.float32)
    vol_in = torch.Tensor(vol_in[None]).to(device)
    
    # ------ initialize model ------ 
    print('Initalize model ...')
    model_white = TANet(
        C_in=1, C_hid=[16,32,64,128,128], inshape=[112,224,176],
        step_size=step_size, M=M, R=R, device=device)
    model_pial = TANet(
        C_in=1, C_hid=[16,32,32,32,32], inshape=[112,224,176],
        step_size=step_size, M=M, R=R, device=device)
    model_white.load_state_dict(torch.load(
        model_dir+data_type+'/model_'+data_type+'_hemi-'+surf_hemi+'_white.pt',
        map_location=device))
    model_pial.load_state_dict(torch.load(
        model_dir+data_type+'/model_'+data_type+'_hemi-'+surf_hemi+'_pial.pt',
        map_location=device))

    # ------ inference ------ 
    print('Start surface reconstruction ...')
    t_start = time.time()
    with torch.no_grad():
        vert_white = model_white(vert_in, vol_in)
        vert_white = taubin_smooth(vert_white, face_in, n_iters=10)
        vert_pial = model_pial(vert_white, vol_in)
    t_end = time.time()
    print('Finished. Runtime:{}'.format(np.round(t_end-t_start,4)))
    
    print('Save surface meshes ...', end=' ')
    # tensor to numpy
    vert_white = vert_white[0].cpu().numpy()
    vert_pial = vert_pial[0].cpu().numpy()
    face_in = face_in[0].cpu().numpy()

    # map surfaces to their original spaces
    vert_white[:,0] = vert_white[:,0] + 64
    vert_white = apply_affine_mat(vert_white, affine_in)
    vert_pial[:,0] = vert_pial[:,0] + 64
    vert_pial = apply_affine_mat(vert_pial, affine_in)
    
    save_gifti_surface(
        vert_white, face_in,
        save_dir+data_type+'_hemi-'+surf_hemi+'_white.surf.gii',
        surf_hemi='CortexLeft', surf_type='GrayWhite')
    save_gifti_surface(
        vert_pial, face_in,
        save_dir+data_type+'_hemi-'+surf_hemi+'_pial.surf.gii',
        surf_hemi='CortexLeft', surf_type='Pial')
    
    print('Done.')
          