import ants
import numpy as np
import cv2
import os, sys, glob, json, shutil

from tqdm import tqdm


def main(initial_vol, label_vol, reference_vol, num_iterations, padding):

    assert padding is None or padding % 2 == 0, 'ValueError: padding should be divisible by 2'

    # set up
    outpath = os.path.dirname(os.path.abspath(initial_vol))
    print('saving outputs to {:}/'.format(outpath))
    print('')
    args_dict = dict(zip(['initial_image', 'reference_image', 'num_iterations', 'padding'], [initial_vol, reference_vol, num_iterations, padding]))
    with open('{:}/2D_to_3D_args.txt'.format(outpath), 'w') as f:
        json.dump(args_dict, f, indent=2)

    # load hist and mri vols
    # requires a reference_vol - MRI volume transformed to Nissl contrast
    hist00 = ants.image_read(initial_vol)
    mri = ants.image_read(reference_vol)
    labels = ants.image_read(label_vol)

    # pad?
    if padding:
        print('padding histology volume by {:} slices at each end'.format(padding//2))
        hist_shape = hist00.shape
        # pad end dimensions
        hist0 = ants.pad_image(hist00, shape=(hist_shape[0], hist_shape[1]+padding, hist_shape[2]))
    else:
        hist0 = hist00.copy()

    # begin iterations
    if padding:
    	file_exists = os.path.isfile('{:}/aligned_MRI_iteration{:02d}_padding{:}.nii.gz'.format(outpath, num_iterations, padding))
    else:
        file_exists = os.path.isfile('{:}/aligned_slices_iteration{:02d}_nopadding.nii.gz'.format(outpath, num_iterations))

    if file_exists is False:
        for it in tqdm(range(num_iterations), desc='iterations'):

            # 3d affine: mri->histology
            print('')
            print('running 3D registration')
            vol_reg = ants.registration(hist0, mri, type_of_transform='Affine')
            # save aligned MRI vol
            if padding:
                outfile = '{:}/aligned_MRI_iteration{:02d}_padding{:}.nii.gz'.format(outpath, it+1, padding)
                vol_reg_shape = vol_reg['warpedmovout'].shape
                # remove end padding
                unpad = vol_reg['warpedmovout'][:, (padding // 2):(vol_reg_shape[1]-(padding // 2)), :]
                unpad = ants.from_numpy(unpad, spacing=hist0.spacing,
                               origin=hist0.origin, direction=hist0.direction)
                ants.image_write(unpad, outfile)
            else:
                outfile = '{:}/aligned_MRI_iteration{:02d}_padding{:}.nii.gz'.format(outpath, it+1, padding)
                ants.image_write(vol_reg['warpedmovout'], outfile)

            # 2D affine: histology -> mri
            hist1, transforms = slicewise_reg(hist0, vol_reg['warpedmovout'])

            # remove end pads and save out
            if padding:
                outfile = '{:}/aligned_slices_iteration{:02d}_padded{:02d}.nii.gz'.format(outpath, it+1, padding)
                unpad = hist1[:, (padding // 2):(hist1.shape[1]-(padding // 2)), :]
                unpad = ants.from_numpy(unpad, spacing=hist1.spacing,
                               origin=hist1.origin, direction=hist1.direction)
                ants.image_write(unpad, outfile)
            else:
                outfile = '{:}/aligned_slices_iteration{:02d}_nopadding.nii.gz'.format(outpath, it+1)
                ants.image_write(hist1, outfile)

            hist0=hist1

    else:
        print('iterations already run - performing final run')


    # run final set of 2D rigid alignments between initial slices and aligned slices
    # pad in I-S direction to allow for drift occuring during iterative regsitration
    if padding:
        unpad = ants.image_read('{:}/aligned_slices_iteration{:02d}_padded{:02d}.nii.gz'.format(outpath, num_iterations, padding))
        padded = ants.pad_image(unpad, shape=(unpad.shape[0], unpad.shape[1], unpad.shape[2]+500))
        aligned_init, transforms = slicewise_reg(hist00, padded, metric='GC', sampling=4)
        outfile = '{:}/final_aligned_slices_iterations{:02d}_padding{:02d}.nii.gz'.format(outpath, num_iterations, padding)

    else:
        unpad = ants.image_read('{:}/aligned_slices_iteration{:02d}_nopadding.nii.gz'.format(outpath, num_iterations))
        padded = ants.pad_image(unpad, shape=(unpad.shape[0], unpad.shape[1], unpad.shape[2]+500))
        aligned_init, transforms = slicewise_reg(hist00, padded, metric='GC', sampling=4)
        outfile = '{:}/final_aligned_slices_iterations{:02d}_nopadding.nii.gz'.format(outpath, num_iterations)

    # save out final slice alignments
    ants.image_write(aligned_init, outfile)

    os.makedirs('{:}/2D_to_3D_transforms'.format(outpath), exist_ok=True)
    # final slicewise between hist00 and hist1
    # apply to label images
    transformed_labels = ants.image_clone(padded)

    for n,tx in enumerate(transforms):
        transformed_labels[:,n,:] = ants.apply_transforms(ants.from_numpy(padded[:,n,:]), ants.from_numpy(labels[:,n,:]), tx[0], interpolator='genericLabel').numpy()
        shutil.copy(tx[0], '{:}/2D_to_3D_transforms/slice{:03d}_to_3D.mat'.format(outpath, n))

    if padding:
        label_outfile = '{:}/final_aligned_labels_iterations{:02d}_padding{:02d}.nii.gz'.format(outpath, num_iterations, padding)
    else:
        label_outfile = '{:}/final_aligned_labels_iterations{:02d}_nopadding.nii.gz'.format(outpath, num_iterations)
    ants.image_write(transformed_labels, label_outfile)

    print('see {:} for final aligned volume'.format(outfile))
    print('see {:} for final aligned labels'.format(label_outfile))

    print('see {:}/2D_to_3D_transforms/ for slice-wise transforms'.format(outpath))



def slicewise_reg(moving_vol, fixed_vol, metric='mattes', sampling=32):
    """
    perform 2d affine registration between corresponding slices in
    moving and fixed volumes

    moving_vol, fixed_vol: 3D ANTsImages

    returns:
    aligned_vol: ANTsImage, moving volume with aligned slices
    transform_list: list, list of ANTs transform for each (non-zero) slice
    """

    aligned_vol = ants.image_clone(fixed_vol)

    num_slices = aligned_vol.shape[1]
    transform_list = []

    for s in tqdm(range(num_slices), desc = 'slicewise registration'):
        fixed_slice = ants.from_numpy(fixed_vol[:,s,:])
        moving_slice = ants.from_numpy(moving_vol[:,s,:])
        moving_mask = ants.get_mask(moving_slice, low_thresh=1, high_thresh=999, cleanup=0)

        if moving_mask.sum() == 0:
            # moving slice is empty
            aligned_vol[:,s,:] = ants.resample_image(moving_slice, (fixed_slice.shape), use_voxels=True).numpy()
        else:
            slice_reg = ants.registration(fixed_slice, moving_slice, type_of_transform='Affine', reg_iterations=(500,500,500,500), aff_metric=metric, sampling=sampling)
            aligned_vol[:,s,:] = slice_reg['warpedmovout'].numpy()
            transform_list.append(slice_reg['fwdtransforms'])

    return aligned_vol, transform_list


if __name__ == '__main__':
    import argparse, os

    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser(description='Perform iterative 2D-to-3D alignment using 3D shape prior')

    parser.add_argument('-i', '--initial_volume', type=file_path, required=True,
            help='initial volume of aligned 2D slices (.nii.gz)')
    parser.add_argument('-l', '--label_volume', type=file_path, required=True,
            help='initial volume of aligned 2D label images (.nii.gz)')
    parser.add_argument('-r', '--reference_volume', type=file_path, required=True,
            help='reference 3D volume (.nii.gz)')
    parser.add_argument('-n', '--number_iterations', type=int, required=True,
                help='number of iterations to perform')
    parser.add_argument('-p', '--padding', type=int, required=False,
                help='number of slices to pad image in A-P direction')
    args = parser.parse_args()

    main(initial_vol = args.initial_volume, label_vol=args.label_volume, reference_vol=args.reference_volume, num_iterations=args.number_iterations, padding=args.padding)
