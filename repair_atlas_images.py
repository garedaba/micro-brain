import glob, json, os, sys

import numpy as np
import cv2

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

sys.path.append('pix2pix')
from utils import get_res_and_size, rgb_to_labels
from models import mosaic_prediction


def main(model_dir = '.', model_to_run=100000,  mad_threshold=2.5):
    if mad_threshold is None:
        mad_threshold = 2.5

    # load trained model - EDIT LOCATION AND NAME OF MODEL AS NEEDED
    trained_model_file = '{:}/model_{:06d}.h5'.format(model_dir, model_to_run)
    print('')
    print('using pretrained model: {:}/model_{:06d}.h5'.format(model_dir, model_to_run))
    print('')
    assert os.path.isfile(trained_model_file), 'NameError: model does not exist'

    trained_model = load_model(trained_model_file)

    # get command line arguments
    argfile = '{:}/commandline_args.txt'.format(model_dir)
    assert os.path.isfile(argfile), 'NameError: command line args do not exist'

    with open(argfile) as f:
        argdict = json.load(f)

    # get image size and resolution of training data
    size, res = get_res_and_size(argdict['train_dir'])

    # get list of preprocessed images to repair
    # EDIT LOCATION AND NAME OF DOWNSAMPLED HISTOLOGY IMAGES AS NEEDED
    nissl_list = sorted(glob.glob('ProcessedImages/original_2um_to_{:03d}um/nissl*masked*'.format(res)))
    label_list = sorted(glob.glob('ProcessedImages/original_2um_to_{:03d}um/label*'.format(res)))
    mask_list = sorted(glob.glob('ProcessedImages/original_2um_to_{:03d}um/mask*'.format(res)))
    assert len(nissl_list) == len(label_list) == len(mask_list), 'ValueError, length of image lists do not match'

    # mkdir directory for output
    label_type = 'rgb'
    if argdict['rgb_to_labels']:
        label_type = 'classlabels'
    outdir = 'RepairedImages/original_2um_to_{:03d}um/repaired_model_res{:}_size{:}_depth{:02d}_batch{:02d}_epochs{:03d}_lr1_{:}_lr2_{:}_lossweight{:03d}_{:}_madthreshold_{:}'.format(res, res, size, argdict['depth'], argdict['batch_num'], argdict['number_epochs'],  argdict['learning_rate1'], argdict['learning_rate2'], argdict['loss_ratio'], label_type, mad_threshold)
    os.makedirs(outdir, exist_ok=True)

    # repair and save out
    for n, img in  enumerate(nissl_list):
        slice_id = img.split('.')[-2]
        print('repairing slice: {:}'.format(slice_id))

        nissl_image = cv2.imread(nissl_list[n], cv2.IMREAD_COLOR)
        label_image = cv2.imread(label_list[n], cv2.IMREAD_COLOR)
        mask_image = cv2.imread(mask_list[n], cv2.IMREAD_GRAYSCALE)
        mask_image = mask_image // 255

        if argdict['rgb_to_labels']:
            test_labels = rgb_to_labels(label_image)
        else:
            test_labels = (label_image - 127.5) / 127.5

        # get model prediction based on overlapping patches
        prediction = mosaic_prediction(test_labels, trained_model, (size, size), batch_size=32, overlap_factor=8, progress_bar=True)
        masked_prediction = (prediction * np.repeat(mask_image[:,:,np.newaxis], 3, axis=2)).astype('uint8')

        # detect outliers
        outliers = detect_label_outliers(nissl_image, masked_prediction, mask_image, mad_threshold=mad_threshold)

        # repair image
        #masked_prediction = (prediction * np.repeat(mask_image[:,:,np.newaxis], 3, axis=2)).astype('uint8')
        repaired = repair_image(nissl_image, masked_prediction, outliers)

        # save out
        cv2.imwrite('{:}/model_prediction.{:}.png'.format(outdir, slice_id), masked_prediction)
        cv2.imwrite('{:}/repaired_section.{:}.png'.format(outdir, slice_id), repaired)
        cv2.imwrite('{:}/outlier_mask.{:}.png'.format(outdir, slice_id), outliers)

    # check if start and end slices are available
    # MANUALLY DRAWN FRONTAL AND OCCIPITAL POLE SECTIONS WERE ADDED TO THE IMAGE STACK
    extras = []
    if glob.glob('ProcessedImages/startslice_*_{:03d}um.png'.format(res)):
        for n,g in enumerate(sorted(glob.glob('ProcessedImages/startslice_*_{:03d}um.png'.format(res)))):
            extras.append([g, '{:}/predicted_start_section_{:02d}.png'.format(outdir, n+1)])
    if glob.glob('ProcessedImages/endslice_*_{:03d}um.png'.format(res)):
        for n,g in enumerate(sorted(glob.glob('ProcessedImages/endslice_*_{:03d}um.png'.format(res)))):
            extras.append([g, '{:}/predicted_end_section_{:02d}.png'.format(outdir, n+1)])

    for image_in, image_out in extras:
        print('adding extras')
        label_image = cv2.imread(image_in, cv2.IMREAD_COLOR)
        mask_image = (label_image.sum(2) >0).astype('uint8')

        if argdict['rgb_to_labels']:
            test_labels = rgb_to_labels(label_image)
        else:
            test_labels = (label_image - 127.5) / 127.5

        # get model prediction based on overlapping patches
        prediction = mosaic_prediction(test_labels, trained_model, (size, size), batch_size=32, overlap_factor=8, progress_bar=True)
        masked_prediction = (prediction * np.repeat(mask_image[:,:,np.newaxis], 3, axis=2)).astype('uint8')

        cv2.imwrite(image_out, masked_prediction)


def detect_label_outliers(image, prediction_image, mask_image, mad_threshold=2.5):
    """
    use hue and saturation measures to detect outlier pixels in regions with
    large colour + contrast differences based on median absolute deviation within brainmask

    image: numpy array (n_pix, n_pix, rgb_channels) dtype(uint8)
    prediction_image:  numpy array (n_pix, n_pix, rgb_channels) dtype(uint8), output from model prediction
    mask_image: numpy array (n_px, n_pix) dtype(uint8), binary (0,1s) image of brain mask
    mad_threshold: threshold to apply to label outliers

    returns:
    outliers: numpy array 0-255 uint8
    """
    assert image.dtype == 'uint8' and prediction_image.dtype == 'uint8' and mask_image.dtype == 'uint8', 'TypeError: uint8 expected'

    # for each label, detect outlying pixels based on hue and saturation
    outliers = get_outliers(image, prediction_image, mask_image, mad_threshold=mad_threshold)

    # any pixels close to white added in
    outliers = (outliers>0).astype('uint8') * 255

    return outliers

def get_outliers(image, prediction_image, mask_image, mad_threshold=2.5):
    """
    detect outliers within a mask region

    image: numpy array (n_pix, n_pix, rgb_channels) dtype(uint8)
    prediction_image:  numpy array (n_pix, n_pix, rgb_channels) dtype(uint8), output from model prediction
    mask_image: numpy array (n_px, n_pix) dtype(uint8), 1s inside mask, 0s outside
    mad_threshold: threshold to apply to label outliers

    returns:
    outliers_mask: outlying pixels, (n_pix, n_pix) uint8
    """
    constant = 1.4826

    # convert to HSV and blur to remove small/noisy differences
    hsv_img = cv2.blur(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (5,5))
    hsv_pred = cv2.blur(cv2.cvtColor(prediction_image, cv2.COLOR_BGR2HSV), (5,5))

    # get indices of pixels in mask
    mask_indices = np.where(mask_image==1)

    # median absolute difference in hue between ground truth and prediction
    diff_h = (np.abs(hsv_img[:,:,0].astype(int) - hsv_pred[:,:,0].astype(int)))
    diff_h_m = np.median(diff_h[mask_indices[0], mask_indices[1]])

    # median absolute difference in saturation between ground truth and prediction
    diff_s = (np.abs(hsv_img[:,:,1].astype(int) - hsv_pred[:,:,1].astype(int)))
    diff_s_m = np.median(diff_s[mask_indices[0], mask_indices[1]])

    # threshold for outliers
    thr_h = diff_h_m * mad_threshold * constant
    thr_s = diff_s_m * mad_threshold * constant

    # get outlier pixels in mask based on hue and sat
    outliers_mask = ((diff_h > thr_h) + (diff_s > thr_s))>0
    outliers_mask = outliers_mask * 255 * mask_image
    outliers_mask = outliers_mask.astype('uint8')

    # remove speckles
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    #outliers_mask = cv2.morphologyEx(outliers_mask, cv2.MORPH_CLOSE, kernel)
    outliers_mask = cv2.morphologyEx(outliers_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    return outliers_mask

def repair_image(original_image, prediction_image, outliers_mask):
    """
    Poisson image editing to repair outlier regions
    http://www.irisa.fr/vista/Papers/2003_siggraph_perez.pdf
    """
    assert original_image.dtype == prediction_image.dtype == outliers_mask.dtype == 'uint8', 'TypeError: uint8 expected'
    assert original_image.shape[2] == prediction_image.shape[2] == 3, 'ValueError: 3-channel RGB expected'

    image_shape = original_image.shape
    centrepoint = (image_shape[1] // 2, image_shape[0] // 2)

    # clones original image (without outliers) on top of predicted image. Outlier regions are left behind
    # and replaced with intensity matched pixels from the prediction image
    image_clone = cv2.seamlessClone(original_image, prediction_image, 255-outliers_mask, centrepoint, cv2.NORMAL_CLONE)

    return image_clone

if __name__ == '__main__':
    import argparse, os

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(description='Run validation of pix2pix model predictions')

    parser.add_argument('-m', '--model_dir', type=dir_path, default='.', required=True,
            help='location of trained model data')
    parser.add_argument('-s', '--model_step', type=int, required=True,
            help='model steps to use')
    parser.add_argument('-t', '--outlier_threshold', type=float, required=False,
                help='outlier threshold to apply (default=2.5)')
    args = parser.parse_args()

    main(model_dir = args.model_dir, model_to_run=args.model_step, mad_threshold = args.outlier_threshold)
