import os
import glob
import time
import numpy as np
from tqdm import tqdm 
import argparse
import shutil
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import affine_transform


if __name__ == "__main__":

    # ------ load arguments ------ 
    parser = argparse.ArgumentParser(description="CoSeg")

    parser.add_argument('--orig_dir', default='YOUR_HCP_DATA/', type=str, help="directory of the original HCP data")
    parser.add_argument('--save_dir', default='./data/hcp/', type=str, help="directory to save the preprocessed data")
    args = parser.parse_args()

    orig_dir = args.orig_dir
    save_dir = args.save_dir

    subj_list = sorted(glob.glob(orig_dir+'*/T1w_restore.nii.gz'))


    # ------ resample HCP data ------ 
    for subj_t1_dir in tqdm(subj_list[:]):
        subj_dir = '/'.join(subj_t1_dir.split('/')[:-1]) + '/'
        subj_id = subj_dir.split('/')[-2]
        print(subj_id)

        if not os.path.exists(save_dir+subj_id):
            os.makedirs(save_dir+subj_id)

        # load image and cortical ribbon segmentation
        img_orig_nib = nib.load(subj_dir+'T1w_restore.nii.gz')
        ribbon_orig_nib = nib.load(subj_dir+'ribbon.nii.gz')

        # resample HCP data to 1mm resolution
        affine_orig = img_orig_nib.affine
        affine_resample = affine_orig.copy()
        affine_resample[0,0] = -1
        affine_resample[1,1] = 1
        affine_resample[2,2] = 1
        affine_resample[0,-1] -= 3.5
        affine_resample[1,-1] -= 1
        affine_resample[2,-1] += 3.5

        # (183, 218, 183) - > ([176,224,176])
        # for image and segmentations we use default resampling
        img_resample_nib = resample_from_to(
            img_orig_nib, ([176,224,176], affine_resample), order=2)
        nib.save(
            img_resample_nib, save_dir+subj_id+'/'+subj_id+'.T1w_restore.nii.gz')

        # for ribbon and brainmask we use multi-channel approaches
        ribbon_orig = ribbon_orig_nib.get_fdata()
        ribbon_onehot = np.zeros_like(ribbon_orig)[None].repeat(5, axis=0)
        ribbon_onehot[0][np.where(ribbon_orig==0)] = 1
        ribbon_onehot[1][np.where(ribbon_orig==2)] = 1
        ribbon_onehot[2][np.where(ribbon_orig==41)] = 1
        ribbon_onehot[3][np.where(ribbon_orig==3)] = 1
        ribbon_onehot[4][np.where(ribbon_orig==42)] = 1

        affine_ = np.linalg.inv(affine_orig).dot(affine_resample)
        affine_mat = affine_[:3,:3]
        affine_off = affine_[:3,3]

        ribbon_resample = np.zeros([5,176,224,176])
        for k in range(5):
            ribbon_resample[k] = affine_transform(
                ribbon_onehot[k], matrix=affine_mat, offset=affine_off,
                output_shape=[176,224,176], order=2)
        ribbon_resample = ribbon_resample.argmax(0)

        ribbon_resample_nib = nib.Nifti1Image(
            ribbon_resample, img_resample_nib.affine, img_resample_nib.header)
        nib.save(
            ribbon_resample_nib, save_dir+subj_id+'/'+subj_id+'.ribbon.nii.gz')

    
    # ------ randomly split train/valid/test data ------
    np.random.seed(12345)
    subj_list = sorted(glob.glob(save_dir+'*/*.T1w_restore.nii.gz'))
    subj_permute = np.random.permutation(len(subj_list))
    n_train = int(len(subj_list) * 0.6)
    n_valid = int(len(subj_list) * 0.1)
    n_test = len(subj_list) - n_train - n_valid
    print('Number of training data:', n_train)
    print('Number of validation data:', n_valid)
    print('Number of testing data:', n_test)

    train_list = subj_permute[:n_train]
    valid_list = subj_permute[n_train:n_train+n_valid]
    test_list = subj_permute[n_train+n_valid:]
    data_list = [train_list, valid_list, test_list]
    data_split = ['train', 'valid', 'test']

    for n in range(3):
        if not os.path.exists(save_dir+data_split[n]):
            os.makedirs(save_dir+data_split[n])
        for i in data_list[n]:
            subj_dir = '/'.join(subj_list[i].split('/')[:-1]) + '/'
            subj_id = subj_dir.split('/')[-2]
            shutil.move(subj_dir, save_dir+data_split[n]+'/'+subj_id)

    print('Done.')