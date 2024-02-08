import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os, sys

from tensorflow.keras.applications.vgg19 import preprocess_input

from numpy.random import randint
from random import randrange


def load_dataset(filename):
    """
    load image and labels patches from pickle

    filename: path to pickled data
    """
    # load data
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 1st and 2nd column are images and labels
    image, label = np.asarray(data[0]), np.asarray(data[1])

    return [image, label]

def generate_real_samples(dataset, n_batch, patch_shape, jitter=True, labels=True):
    """
    generate pair of image and labels with 'true' labels for discriminator

    dataset: image/label pairs
    n_batch: number of patches in each batch
    patch_shape: size of patchGAN output
    jitter: add random jitter and crop for augmentation
    labels: convert 3 channel RGB label image to n_labels-channel image

    returns:
    [X1, X2]: list of paired, processed image pairs
    y: patch_shape x patch_shape of ones
    """
    # unpack dataset
    train_image, train_label = dataset

    # choose random selection for batch
    ix = randint(0, train_image.shape[0], n_batch)

    # retrieve selected images
    X1, X2 = train_image[ix], train_label[ix]

    # random jitter and horizontal/vertical flip
    if jitter is True:
        for n, img in enumerate(X1):
            X1[n], X2[n] = jitter_samples(X1[n], X2[n], upsize=356)

    # transform rgb to class labels (one channel per label)
    if labels is True:
        X1 = (X1 - 127.5) / 127.5 #transform nissl to [-1,1]
        X2 =  np.asarray([rgb_to_labels(n) for n in X2]) # labels are [-1,1]
    else:
        X1 = (X1 - 127.5) / 127.5 #transform nissl to [-1,1]
        X2 = (X2 - 127.5) / 127.5 #transform rgb labels to [-1,1]

    # generate clf labels for 'real' nissl images (all 1s for patchGan)
    y = np.ones((n_batch, patch_shape, patch_shape, 1)) * .9 # label smoothing

    return [X1, X2], y


def jitter_samples(img, targ_img, upsize=300):
    """
    apply random crop and jitter as well as random flips for data augmentation

    img: image patch
    targ_img: label patch
    upsize: size to expand patch to before crop

    returns:
    im: jittered, cropped, flipped image patch
    targ_im: jittered, cropped, flipped label patch
    """
    img_shape = img.shape
    # convert to Image
    im = Image.fromarray((img).astype(np.uint8))
    targ_im = Image.fromarray((targ_img).astype(np.uint8))

    # resize
    im = im.resize((upsize,upsize), Image.NEAREST)
    targ_im = targ_im.resize((upsize,upsize), Image.NEAREST)

    # random crop
    xmax = ymax = upsize-img_shape[0]
    random_x = randrange(0, xmax//2 + 1) * 2
    random_y = randrange(0, ymax//2 + 1) * 2
    area = (random_x, random_y, random_x + img_shape[0], random_y + img_shape[0])
    im = im.crop(area)
    targ_im = targ_im.crop(area)

    # flip
    if np.random.rand()>.5:
        im.transpose(Image.FLIP_LEFT_RIGHT)
        targ_im.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand()>.5:
        im.transpose(Image.FLIP_TOP_BOTTOM)
        targ_im.transpose(Image.FLIP_TOP_BOTTOM)

    return np.array(im), np.array(targ_im)


def generate_fake_samples(g_model, samples, patch_shape):
    """
    generates synthetic samples for discriminator

    g_model: generator model
    samples: label patch to transform
    patch_shape: size of patchGAN output

    returns:
    X: sythetic image patch
    y: patch_shape x patch_shape of zeros
    """
    # generate fake instance
    X = g_model.predict(samples)

    # create clf labels for 'fake' nissl images (all 0s)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))

    return X, y


def summarise_performance(step, g_model, dataset, labels=False, n_samples=3, savepath='.'):
    # https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
    """
    generates synthetic image from generator, compares to true image and saves images and model

    step: model step number
    g_model: trained generator
    dataset: paired image/label patches
    labels: False=RGB channel labels, True=n_label channels
    n_samples: number of samples to generate
    savepath: path to directory to save model and images

    returns:
    """
    # select a sample of input images
    [X_real_image, X_real_label], _ = generate_real_samples(dataset, n_samples, 1, labels=labels)

    # generate a batch of fake images from real labels
    X_fake_image, _ = generate_fake_samples(g_model, X_real_label, 1)

    # if label channels are used, transform back to RGB and [0,1]
    if labels is True:
        X_real_label =  np.asarray([labels_to_rgb(n) for n in X_real_label]) # labels to rgb channels
        X_real_label = X_real_label / 255

    else:
        # scale all label pixels from [-1,1] to [0,1]
        X_real_label = (X_real_label + 1) / 2.0

    #scale images to [0,1]
    X_real_image = (X_real_image + 1) / 2.0
    X_fake_image = (X_fake_image + 1) / 2.0

    # create output figure
    plt.figure(figsize=(15,10))
    # plot real target images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_real_label[i][:,:,::-1])
    # plot generated nissl image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fake_image[i][:,:,::-1])
    # plot real nissl image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_real_image[i][:,:,::-1])

    # save plot to file
    filename1 = savepath+'/plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()

    # save the generator model
    filename2 = savepath+'/model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def get_lut(cortex=False):
    """
    generates a LUT dictionary from label_lut.csv

    returns:
    lut_dict: dictionary of label -> rgb
    """
    filedir = (os.path.dirname(os.path.abspath(__file__)))
    if cortex:
        lut_table = pd.read_csv('{:}/cortical_label_lut.csv'.format(filedir))
    else:
        lut_table = pd.read_csv('{:}/label_lut.csv'.format(filedir))

    # images loaded in as bgr
    lut_dict = dict(zip(lut_table['label_id'], lut_table[['B','G','R']].values))

    return lut_dict

def rgb_to_labels(rgb_patch, cortex=False):
    """
    transform 3-channel BGR to n_label-channel image (one channel per label class)

    rgb_patch: label patch with BGR channels

    returns:
    labels_classes: image with n_label channels, each scaled to [-1,1]
    """
    lut_dict = get_lut(cortex=cortex)

    patchdim = rgb_patch.shape
    label_classes = np.zeros((patchdim[0], patchdim[1], len(lut_dict)), int)

    # n_pix, n_pix, n_labels
    for i in np.arange(len(lut_dict)):
        #[-1,1] for each class
        label_classes[:,:,i] = (2*(rgb_patch==lut_dict[i]).all(2)) - 1

    return label_classes

def labels_to_rgb(class_image, cortex=False):
    """
    transform n_label-channel to 3-channel BGR

    class_image: label patch with n_label channels

    returns:
    rgb_image: image with 3-channel BGR
    """

    lut_dict = get_lut(cortex=cortex)

    patchdim = class_image.shape
    rgb_image = np.zeros((patchdim[0], patchdim[1], 3), int)

    # n_pix, n_pix, 3
    for i in np.arange(len(lut_dict)):
        rgb_image = np.where((class_image[:,:,i]==1)[:,:,np.newaxis], lut_dict[i], rgb_image)

    return rgb_image

def get_res_and_size(path):
    """
    get resolution and patch size from directory name

    path: path to directory

    returns:
    patch_size, patch_resolution
    """
    dirsplit = path.split('_')

    sz = [x for x in dirsplit if 'size' in x]
    assert len(sz)>0, 'training directory error'

    patch_size = int(''.join(i for i in sz[0] if i.isdigit()))

    rs = [x for x in dirsplit if 'res' in x]
    assert len(rs)>0, 'training directory error'

    patch_resolution = int(''.join(i for i in rs[0] if i.isdigit()))

    return patch_size, patch_resolution

def centre_pad(image, patch_size):
    """
    add border to image to ensure equal split into patches pf patch_size

    image: array, nxn greyscale image
    patch size: length in pixels of patch

    returns:
    result: padded image
    """
    if np.ndim(image)==3:
        w, h, d = image.shape
    else:
        w, h = image.shape

    if w % patch_size == 0:
        pad_w = patch_size*((w//patch_size))
    else:
        pad_w = patch_size*((w//patch_size)+1)

    if h % patch_size == 0:
        pad_h = patch_size*((h//patch_size))
    else:
        pad_h = patch_size*((h//patch_size)+1)

    # compute center offset
    ww = (pad_w - w) // 2
    hh = (pad_h - h) // 2

    # copy img image into center of result image
    if np.ndim(image)==3:
        result = np.full((pad_w, pad_h, d), fill_value=0, dtype=np.uint8)
        result[ww:ww+w, hh:hh+h, :] = image
    else:
        result = np.full((pad_w, pad_h), fill_value=0, dtype=np.uint8)
        result[ww:ww+w, hh:hh+h] = image

    return result

def split_patches(image, patch_size):
    """
    split image into a set of patches of size (patch_size x patch_size)

    image: image to split
    patch_size: size of patch (assumed square)

    returns:
    blocks: list of patches of size (patch_size x patch_size)
    """
    assert image.shape[0] % patch_size ==0, 'image dimensions dont match patch'
    assert image.shape[1] % patch_size ==0, 'image dimensions dont match patch'

    blocks = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0,image.shape[1],patch_size):
            blocks.append(image[i:i+patch_size, j:j+patch_size])

    return blocks

def checkerboard_split(image, patch_size):
    """
    split images into checkerboard pattern for validation

    image: image to split
    patch_size: size of patch (assumed square)

    returns:
    valid_blocks, test_blocks: lists of alternating patches in checkerboard pattern
    """
    assert image.shape[0] % patch_size ==0, 'image dimensions dont match patch'
    assert image.shape[1] % patch_size ==0, 'image dimensions dont match patch'

    valid_blocks = []
    test_blocks = []

    for ni, i in enumerate(range(0, image.shape[0], patch_size)):
        for nj, j in enumerate(range(0,image.shape[1],patch_size)):
            if ni % 2 == 0:
                if nj % 2 == 0:
                    valid_blocks.append(image[i:i+patch_size, j:j+patch_size])
                else:
                    test_blocks.append(image[i:i+patch_size, j:j+patch_size])
            else:
                if nj % 2 == 0:
                    test_blocks.append(image[i:i+patch_size, j:j+patch_size])
                else:
                    valid_blocks.append(image[i:i+patch_size, j:j+patch_size])

    return valid_blocks, test_blocks


def get_patches(im, patch_size=128):
    """
    perform image padding and checkerboard split

    im: image to split
    patch_size: size of patch (assumed square)

    returns:
    valid_blocks, test_blocks: lists of alternating patches in checkerboard pattern
    im_pad: padded image
    """
    # pad
    im_pad = centre_pad(im, patch_size)

    # split
    valid_im_blocks, test_im_blocks = checkerboard_split(im_pad, patch_size)

    return valid_im_blocks, test_im_blocks, im_pad

def stitch_images(image_width, image_height, valid_patches, test_patches):
    """
    stitch images back together from a set of checkerboard patches

    image_width: width of image to reconstruct in pixels
    image_height: width of image to reconstruct in pixels
    valid_patches, test_patches: lists of alternating checkerboard patches

    returns:
    image: stitched image
    """

    patch_size = np.shape(valid_patches)[1]
    num_patches = np.shape(valid_patches)[0] + np.shape(test_patches)[0]

    assert image_width % patch_size == 0, 'image dimensions not valid'
    assert image_height % patch_size == 0, 'image dimensions not valid'
    assert (image_width // patch_size) * (image_height // patch_size) == num_patches, 'image dimensions not valid'

    image = np.zeros((image_height, image_width, 3), dtype=int)

    counter = 0
    for ni,i in enumerate(range(0, image_height, patch_size)):
        for nj, j in enumerate(range(0, image_width, patch_size)):
            if ni % 2 == 0:
                if nj % 2 == 0:
                    image[i:i+patch_size, j:j+patch_size, :] = valid_patches[counter // 2]
                else:
                    image[i:i+patch_size, j:j+patch_size, :] = test_patches[(counter // 2)]
            else:
                if nj % 2 == 0:
                    image[i:i+patch_size, j:j+patch_size, :] = test_patches[(counter // 2)]
                else:
                    image[i:i+patch_size, j:j+patch_size, :] = valid_patches[(counter // 2)]
            counter += 1

    return image

def get_vgg_features(image, model):
    """
    extract high-level features from VGG model

    image: image patch
    model: pre-trained VGG model

    return:
    vector of image features
    """
    image_proc = preprocess_input(np.expand_dims(image, 0))
    return model.predict(image_proc).flatten()

def get_perceptual_sim(image1, image2, model, bgr=True):
    """
    calculate similarity of VGG features for two patches

    image1, image2: patches to compare
    model: pretrained VGG model
    bgr=True: True = colour channels are BGR order

    returns:
    percept_sim: image similarity (1 / (1+Euclidean distance))
    """
    if bgr:
        image1 = image1[:,:,::-1]
        image2 = image2[:,:,::-1]

    features1 = get_vgg_features(image1, model)
    features2 = get_vgg_features(image2, model)

    # normalise
    features1 = features1 / np.linalg.norm(features1)
    features2 = features2 / np.linalg.norm(features2)

    # similarity
    percept_sim = 1 / (1 + np.sqrt(np.sum((features1 - features2) ** 2)))

    return percept_sim

def get_hue_sim(image1, image2, mask):
    """
    calculate similarity in hue between two image patches

    image1, image2: patches to compare
    mask: mask within which to calculate similarity

    returns:
    hue_sim: similarity in hue (1/(1+Euclidean distance))
    """

    # convert to hsv
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    mask = mask.flatten().astype(int)
     # get hue and saturation from within brain regions
    hue1 = hsv1[:,:,0].flatten().astype(int)[mask==1]
    hue2 = hsv2[:,:,0].flatten().astype(int)[mask==1]
      # normalise to unit length
    hue1 = hue1 / np.linalg.norm(hue1)
    hue2 = hue2 / np.linalg.norm(hue2)

    # similarity in H & S (1 - Euclidean distance between standardised vectors)
    hue_sim = 1 / (1 + np.sqrt(np.sum((hue1 - hue2) ** 2)))

    return hue_sim


def get_sat_sim(image1, image2, mask):
    """
    calculate similarity in saturation between two image patches

    image1, image2: patches to compare
    mask: mask within which to calculate similarity

    returns:
    sat_sim: average similarity in saturation (1/(1+Euclidean distance))
    """

    # convert to hsv
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    mask = mask.flatten().astype(int)
     # get hue and saturation from within brain regions
    sat1 = hsv1[:,:,1].flatten().astype(int)[mask==1]
    sat2 = hsv2[:,:,1].flatten().astype(int)[mask==1]
      # normalise to unit length
    sat1 = sat1 / np.linalg.norm(sat1)
    sat2 = sat2 / np.linalg.norm(sat2)

    # similarity in H & S (1 - Euclidean distance between standardised vectors)
    sat_sim = 1 / (1 + np.sqrt(np.sum((sat1 - sat2) ** 2)))

    return sat_sim
