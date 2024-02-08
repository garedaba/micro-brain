import ants
import numpy as np
import cv2
import os, sys, glob, json

import networkx as nx
import pandas as pd

from tqdm import tqdm

sys.path.append('pix2pix')
from utils import rgb_to_labels

def main(repair_dir='.', reference_id = 40, neighbours=5, reg=0.5, padding=100):

    # set output directory and save arguments
    os.makedirs('AlignedImages/transforms', exist_ok=True)
    args_dict = dict(zip(['repair_directory', 'reference_section', 'neighbours', 'lambda', 'padding'], [repair_dir, reference_id, neighbours, reg, padding]))
    with open('AlignedImages/commandline_args.txt', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # load slice section data - contains thickness of each section (drop any to be removed)
    section_data = pd.read_csv('AtlasData/slice_data.csv')
    keep_sections = list(section_data['slices'].values > 0)

    # list of repaired images
    repaired_im_list = sorted(glob.glob('{:}/repaired_section*'.format(repair_dir)))
    # only keep those listed above
    repaired_im_list = [repaired_im_list[n] for n, m in enumerate(keep_sections) if m]
    assert reference_id > 0  and reference_id < len(repaired_im_list), "ValueError: reference id shouldn't be first or last section"
    # load as grayscale
    repaired_im_list = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in repaired_im_list]

    # load labels images for right resolution
    rs = [x for x in repair_dir.split('_') if 'res' in x]
    rs = int(''.join(i for i in rs[0] if i.isdigit()))
    label_im_list = sorted(glob.glob('ProcessedImages/original_2um_to_{:03d}um/labels.orig*.new*.png'.format(rs)))
    label_im_list = [label_im_list[n] for n, m in enumerate(keep_sections) if m]
    assert len(repaired_im_list) == len(label_im_list), "Number of label images doesn't match"

    # load as BGR and convert to label images
    label_im_list = [np.argmax(rgb_to_labels(cv2.imread(img, cv2.IMREAD_COLOR)), axis=2).astype('uint8') for img in label_im_list]

    # pad reference image to allow space for mis-registration
    if padding:
        repaired_im_list = [cv2.copyMakeBorder(im, padding, padding, padding, padding, cv2.BORDER_CONSTANT) for im in repaired_im_list]
        label_im_list = [cv2.copyMakeBorder(im, padding, padding, padding, padding, cv2.BORDER_CONSTANT) for im in label_im_list]

    # convert to ANTs format
    repaired_ants_im_list = [ants.from_numpy(img) for img in repaired_im_list]
    label_ants_im_list = [ants.from_numpy(img) for img in label_im_list]

    # calculate transforms and similarities
    transforms, similarities = calculate_transforms(repaired_ants_im_list, neighbours, reference_id, lambda_reg = reg)

    # transform images to reference space via shortest path
    transformed_images = shortest_path_transforms(repaired_ants_im_list, reference_id, transforms, similarities, outdir='AlignedImages/transforms/')

    # transform label data
    transformed_labels = apply_transforms(label_ants_im_list, reference_id, outdir='AlignedImages/transforms/')

    # output to initial volume
    # stack to match thickness
    thicknesses = section_data['slices'].values[keep_sections]
    stacked_slices = np.repeat(transformed_images, thicknesses)
    stacked_labels = np.repeat(transformed_labels, thicknesses)

    # add end sections if available
    if glob.glob('{:}/predicted_start_section_*.png'.format(repair_dir)):
        reg_fixed = transformed_labels[0]
        for n,g in enumerate(sorted(glob.glob('{:}/predicted_start_section_*.png'.format(repair_dir)))):
            label_img = 'ProcessedImages/startslice_{:02d}_{:03d}um.png'.format(n+1, rs)
            start_section = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
            start_label = np.argmax(rgb_to_labels(cv2.imread(label_img, cv2.IMREAD_COLOR)), axis=2).astype('uint8')
            if padding:
                start_section = cv2.copyMakeBorder(start_section, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
                start_label = cv2.copyMakeBorder(start_label, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

            # apply_transforms causing issues, so just run a 1-iteration rigid registration to resample!
            reg = ants.registration(reg_fixed, ants.from_numpy(start_section), type_of_transform='Rigid')
            start_section = reg['warpedmovout']
            start_label = ants.apply_transforms(reg_fixed, ants.from_numpy(start_label), reg['fwdtransforms'], interpolator='genericLabel')

            # add two
            stacked_slices = np.insert(stacked_slices, 0, start_section)
            stacked_slices = np.insert(stacked_slices, 0, start_section)
            stacked_labels = np.insert(stacked_labels, 0, start_label)
            stacked_labels = np.insert(stacked_labels, 0, start_label)

            reg_fixed = start_section


    if glob.glob('{:}/predicted_end_section_*.png'.format(repair_dir)):
        reg_fixed = transformed_labels[-1]
        for n,g in enumerate(sorted(glob.glob('{:}/predicted_end_section_*.png'.format(repair_dir)))):
            label_img = 'ProcessedImages/endslice_{:02d}_{:03d}um.png'.format(n+1, rs)
            end_section = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
            end_label = np.argmax(rgb_to_labels(cv2.imread(label_img, cv2.IMREAD_COLOR)), axis=2).astype('uint8')
            if padding:
                end_section = cv2.copyMakeBorder(end_section, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
                end_label = cv2.copyMakeBorder(end_label, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

            reg = ants.registration(reg_fixed, ants.from_numpy(end_section), type_of_transform='Rigid')
            end_section = reg['warpedmovout']
            end_label = ants.apply_transforms(reg_fixed, ants.from_numpy(end_label), reg['fwdtransforms'], interpolator='genericLabel')

            # add two
            stacked_slices = np.insert(stacked_slices, len(stacked_slices), end_section)
            stacked_slices = np.insert(stacked_slices, len(stacked_slices), end_section)
            stacked_labels = np.insert(stacked_labels, len(stacked_labels), end_label)
            stacked_labels = np.insert(stacked_labels, len(stacked_labels), end_label)

            reg_fixed = end_section

    # output to volume
    stacked_slices = [i * (i>10) for i in stacked_slices] # apply low threshold to remove some noise around edges
    output_nifti_from_stacked_slices(stacked_slices, 'AlignedImages/initial_volume.nii.gz')
    output_nifti_from_stacked_slices(stacked_labels, 'AlignedImages/initial_labels.nii.gz')


def get_slice_pairs(slice_index, epsilon, reference_slice_index):
    """
    # https://github.com/pmajka/poSSum/blob/master/bin/pos_sequential_alignment

    Returns pairs of slices between which partial transformations will be
    calculated to create graph

    slice_index: int, index of section to register
    epsilon: int, number of neighbouring section to calculate transforms to
    reference_slice_index: int, index of the reference section

    returns:
    pair_list: list of tuples, each a pair of sections to calculate transforms between
    """
    start_index = slice_index - epsilon
    end_index = slice_index + epsilon

    # Array holding pairs of transformations between which the
    # transformations will be calculated.
    pair_list = []

    if slice_index == reference_slice_index:
        pair_list.append((slice_index, slice_index))
    if slice_index > reference_slice_index:
        for j in list(range(start_index, slice_index))[::-1]:
            if j <= end_index and slice_index != j and j >= reference_slice_index:
                pair_list.append((slice_index, j))
    if slice_index < reference_slice_index:
        for j in list(range(slice_index, end_index + 1)):
            if j >= start_index and slice_index != j and j <= reference_slice_index:
                pair_list.append((slice_index, j))

    return pair_list

def calculate_transforms(image_list, neighbours, reference_id, lambda_reg = 0.5):
    """
    calculate pairwise transforms between adjacent sections towards a reference section

    image_list: list, list of ANTs images
    neighbours: int, number of neighbouring section to calculate transforms to
    reference_id: int, index of the reference section
    lambda_reg: penalty on skipping slices

    returns:
    transform_array: list of lists, ANTs transforms between adjacent sections
    similarity_array: weighted similarity between transformed sections
    """
    assert type(image_list[0]) == ants.core.ants_image.ANTsImage, "TypeError: image not ANTsImage"

    num_section = len(image_list)

    # array for pairwise transforms
    transform_array = [[0 for i in range(num_section)] for j in range(num_section)]

    # array for image similarities
    similarity_array = [[0 for i in range(num_section)] for j in range(num_section)]

    for section in tqdm(np.arange(num_section), desc='calculating neighbourhood transforms'):
        pairs = get_slice_pairs(section, neighbours, reference_id)

        for p in pairs:
            # calculate rigid transform
            transform = ants.registration(image_list[p[1]], image_list[p[0]], type_of_transform='Rigid', metric='meansquares')
            # correlation between aligned images
            cc = np.corrcoef(image_list[p[1]].numpy().reshape(-1), transform['warpedmovout'].numpy().reshape(-1))[0,1]
            # invert and weight by distance between sections (lower = better alignment)
            # https://github.com/pmajka/poSSum/blob/master/bin/pos_sequential_alignment
            # set neighbours to 1 to just calculate direct path 0->1->2->3->...->ref
            diff = p[0] - p[1]
            weight = (1 - cc) * ((1 + lambda_reg) ** (abs(diff)))

            # store in arrays
            transform_array[p[0]][p[1]] = transform['fwdtransforms'][0]
            similarity_array[p[0]][p[1]] = weight

    return transform_array, similarity_array

def shortest_path_transforms(image_list, reference, transforms, similarities, outdir='AlignedImages/transforms/'):
    """
    use Dijkstra's algorithm to calculate lowest cost transform between source and
    reference sections via shortest path

    image_list: list, list of ANTs images
    reference: int, index of reference section
    transforms: list of lists, pairwise transforms
    similarities: list of lists, pairwise similarities of transformed images

    returns:
    transformed_sections: list, list of transformed ANTsImages
    """
    assert type(image_list[0]) == ants.core.ants_image.ANTsImage, "TypeError: image not ANTsImage"

    num_section = len(image_list)

    # make graph from similarities
    G = nx.from_numpy_array(np.array(similarities))

    transformed_section_list = []

    # transform sections to reference
    for src in tqdm(range(num_section), desc='transforming sections'):
        transform_list = []
        # calculate list of transform along shortest path
        last_step = src
        for step in nx.shortest_path(G, source=src, target=reference, weight='weight')[1:]:
            transform_list.append(transforms[last_step][step])
            last_step = step

        # calculate composite transform and save out
        composed_transform_path = ants.apply_transforms(image_list[reference], image_list[src], transform_list, compose='{:}{:02d}-to-{:02d}-transform-'.format(outdir, src, reference))
        # transform section
        transformed_section = ants.apply_transforms(image_list[reference], image_list[src], composed_transform_path)
        transformed_section_list.append(transformed_section)

        # save out
        cv2.imwrite('{:}{:02d}-to-{:02d}-transformed-section.png'.format(outdir, src, reference), transformed_section.numpy().astype('uint8'))

    return transformed_section_list

def apply_transforms(image_list, reference_id, outdir='AlignedImages/transforms/'):
    """
    takes list of label images and transforms with corresponding transforms for each slice

    image_list: list, list of ANTsImages
    reference_id: int, index of reference slice

    returns:
    transformed_sections: list, list of transformed ANTsImages
    """
    transformed_section_list = []

    for n, im in enumerate(image_list):
        transformed_label_section = ants.apply_transforms(image_list[reference_id], im, '{:}{:02d}-to-{:02d}-transform-comptx.nii.gz'.format(outdir, n, reference_id), interpolator='genericLabel')
        transformed_section_list.append(transformed_label_section)

    return transformed_section_list


def output_nifti_from_stacked_slices(stacked_slices, outfile):
    """
    takes list of aligned sections and outputs as a nifti

    stacked_slices: list, list of aligned ANTsImages
    outfile: path for output

    """
    x, y = stacked_slices[0].shape
    d = len(stacked_slices)

    volume = np.zeros(( y, d, x )) # for correct nifti orientation

    for n,i in enumerate(stacked_slices):
        volume[:,n,:] = i.numpy()[::-1,:].T

    ants_volume = ants.from_numpy(volume, spacing = (.02,.5,.02))

    ants.image_write(ants_volume, filename=outfile)
    print('see: {:}'.format(outfile))



if __name__ == '__main__':
    import argparse, os

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(description='Use graph-based sequential alignment to create initial 3D volume')

    parser.add_argument('-d', '--repair_dir', type=dir_path, default='.', required=True,
            help='location of repaired tissue sections')
    parser.add_argument('-r', '--reference_id', type=int, choices=range(0,81,1), metavar=[0-81], required=True,
            help='which section to use as reference')
    parser.add_argument('-n', '--neighbours', type=int, required=True,
                help='how many neighbours to use for pairwise registration')
    parser.add_argument('-l', '--lambda_reg', type=float, required=True,
                help='regularisation for shortest path calculations - higher weights = less chance of skipping sections')
    parser.add_argument('-p', '--padding', type=int, required=True,
                help='padding in x and y dimension')
    args = parser.parse_args()

    main(repair_dir = args.repair_dir, reference_id=args.reference_id, neighbours=args.neighbours, reg=args.lambda_reg, padding = args.padding)
