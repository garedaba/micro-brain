import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow.keras.backend as K
import tensorflow as tf
from utils import generate_real_samples
from utils import generate_fake_samples
from utils import summarise_performance

from MightyMosaic import MightyMosaic

# perception model
def get_perception_model(image_size, pooling='avg'):
    assert isinstance(image_size, tuple), 'TypeError: check image dimensions'
    assert len(image_size)==3, 'ValueError: check image dimensions'

    # load VGG19 model for feature extraction
    input_tensor = Input(shape=image_size)
    vggmodel = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling=pooling)

    return vggmodel

# predict images using trained model
def mosaic_prediction(labels, model, patch_size, batch_size=8, overlap_factor=2, fill_mode='reflect', progress_bar=True):
    """
    perform image prediction of patches and merge together to full image using MightyMosaic. Although the GAN is trained
    with batch size of 1, we predict in a batch to standardise predicted images to same mean /var as we are stitching together
    neighbouring patches. The BN layers in pix2pix are set to trainable, therefore would use sample mean and var for each
    prediction, which may cause some differences in contrast between outputs of neighbouring patches if predicted independently.
    In practice, batchsize >= 4 seems to produce similar results.

    labels: label image to predict
    model: trained generator model
    patch_size: size of path to predict (same size as model was trained on)
    overlap_factor: overlap of adjacent patches
    fill_mode: points outside the boundaries of the input are filled according to the given mode (see keras.ImageDataGenerator)

    returns:
    merged image prediction
    """
    assert isinstance(patch_size, tuple), 'TypeError: check patch dimensions'

    mosaic = MightyMosaic.from_array(labels, patch_size, overlap_factor=overlap_factor, fill_mode=fill_mode)
    fused_prediction = mosaic.apply(model.predict, batch_size=batch_size, progress_bar=progress_bar)
    predicted = fused_prediction.get_fusion()

    return (((predicted + 1) / 2)*255).astype('int')

#-------DISCRIMINATOR---------------------------------------------------------#
# https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
# https://arxiv.org/pdf/1611.07004.pdf Appendix: 6.1.2
# C64-[C128-C256-C512] with an additional CXXX based on implementation above

def define_discriminator(image_src_shape, image_trg_shape, initial_filters=64, depth=3, lr=0.0002):
    # depth 3 = receptive field of 70
    # depth 2 = receptive field of 34
    # depth 1 = receptive field of 16

    assert depth in [1, 2, 3], 'depth field value not allowed'

    # weight initialization
    init = RandomNormal(stddev=0.02)

    # source image input
    in_src_image = Input(shape=image_src_shape)

    # target image input
    in_target_image = Input(shape=image_trg_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # first layer
    d = Conv2D(initial_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    for i in range(1,depth):
        mult = min(2 ** i, 8)

        d = Conv2D(initial_filters * mult, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

    # 2nd last layer - stride 1
    mult = min(2 ** depth, 8)
    d = Conv2D(initial_filters * mult, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # patch output
    d = Conv2D(1, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    # define model
    model = Model([in_src_image, in_target_image], patch_out)

    # compile model
    opt = Adam(lr=lr, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    return model

#-------GENERATOR--------------------------------------------------------------#

def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)

    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)

    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    #g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = UpSampling2D(interpolation='nearest')(layer_in)
    g = Conv2D(n_filters, (4,4), padding='same', kernel_initializer=init)(g)
    # add batch normalization
    g = BatchNormalization()(g, training=True)

    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)

    # merge with skip connection
    g = Concatenate()([g, skip_in])

    # relu activation
    g = Activation('relu')(g)

    return g


def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # image input
    in_image = Input(shape=image_shape)
    patchsize = image_shape[1]

    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)

    if patchsize==128:
        # finish one layer early
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
        b = Activation('relu')(b)
    else:
        e7 = define_encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)

    # decoder model: CD512-CD1024-CD1024-CD1024-CD1024-C512-C256-C128 w/ dropout @50%
    if patchsize==128:
        d2 = decoder_block(b, e6, 512, dropout=True)
    else:
        d1 = decoder_block(b, e7, 512, dropout=True)
        d2 = decoder_block(d1, e6, 512, dropout=True)

    d3 = decoder_block(d2, e5, 512, dropout=True)
    d4 = decoder_block(d3, e4, 512, dropout=True)
    d5 = decoder_block(d4, e3, 256, dropout=True)
    d6 = decoder_block(d5, e2, 128, dropout=True)
    d7 = decoder_block(d6, e1, 64, dropout=True)

    # output
    g = UpSampling2D(interpolation='nearest')(d7)
    g = Conv2D(3, (4,4), padding='same', kernel_initializer=init)(g)

    out_image = Activation('tanh')(g)

    # define model
    model = Model(in_image, out_image)

    return model

#-------GAN--------------------------------------------------------------------#

def define_gan(g_model, d_model, image_shape, lr=0.0002, loss_ratio=100):

    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    # define the source image
    in_src = Input(shape=image_shape)

    # connect the source image to the generator input
    gen_out = g_model(in_src)

    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])

    # src image as input, [generated image and classification output]
    model = Model(in_src, [dis_out, gen_out])

    # compile model - model upates generator using clf accuracy and mae of gen image
    opt = Adam(lr=lr, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,loss_ratio])

    return model


#-------train------------------------------------------------------------------#
def train(d_model, g_model, gan_model, dataset, jitter=False, labels=False, n_epochs=100, n_batch=1, savepath='.'):

    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # unpack dataset
    train_images, train_labels = dataset

    # calculate the number of batches per training epoch
    bat_per_epo = int(len(train_images) / n_batch)

    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    # save losses
    step=[]
    g_loss_all=[]
    d_loss_true = []
    d_loss_fake = []

    # manually enumerate epochs
    for i in range(n_steps):

        # select a batch of real samples
        [X_real_image, X_real_label], y_real = generate_real_samples(dataset, n_batch, n_patch, jitter=jitter, labels=labels) #jitter?

        # generate a batch of fake samples
        X_fake_image, y_fake = generate_fake_samples(g_model, X_real_label, n_patch)

        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_real_label, X_real_image], y_real)

        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_real_label, X_fake_image], y_fake)

        # update the generator
        g_loss, g_loss_clf, g_loss_mae = gan_model.train_on_batch(X_real_label, [y_real, X_real_image])

        # summarize performance                                      #check echk
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f] [%.3f] [%.3f]' % (i+1, d_loss1, d_loss2, g_loss, g_loss_clf, g_loss_mae))
        step.append(i+1)
        g_loss_all.append(g_loss)
        d_loss_true.append(d_loss1)
        d_loss_fake.append(d_loss2)

        # summarize model performance
        if (i==0) or ((i+1) % (5*(bat_per_epo)) == 0):
            summarise_performance(i, g_model, dataset, savepath=savepath, labels=labels)

    loss_all = pd.DataFrame((step, g_loss_all, d_loss_true, d_loss_fake)).T
    loss_all.columns = ['step', 'g_loss', 'd_loss_real', 'd_loss_fake']

    return loss_all
