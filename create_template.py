import glob, ants, itertools, os, json
import numpy as np

from tempfile import mktemp
from tqdm import tqdm

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '16'

def main(initial_image, label_image, cortical_label_image, num_versions=50, slice_drop=10, slice_neighbours=5, slice_weights='linear', reg_iter=3, tmpdir='/tmp', template_resolution=0.1):
    print('')
    print('beginning registrations with OPTIONS:')
    print("DATA AUGMENTATION")
    print('producing {:} versions (slice drop = {:})'.format(num_versions, slice_drop))
    print('')
    print('INITIAL REGISTRATION - {:} ITERATIONS'.format(reg_iter))
    print('using {:} neighbours with {:} weighting'.format(slice_neighbours, slice_weights))
    print('storing temporary files in {:}'.format(tmpdir))
    print('')
    print('TEMPLATE')
    print('final resolution {:}mm'.format(template_resolution))
    print('')

    # locations
    image_dir = os.path.dirname(os.path.abspath(initial_image))
    stack_dir = image_dir + '/image_stacks'
    os.makedirs(stack_dir, exist_ok = True)

    # save out args
    args_dict = dict(zip(['initial_image', 'label_image', 'cortical_label_image', 'num_volumes', 'slice_drop', 'slice_neighbours', 'slice_weights', 'nonlinear_reg_iterations', 'resolution'], [initial_image, label_image, cortical_label_image, num_versions, slice_drop, slice_neighbours, slice_weights, reg_iter, template_resolution]))
    with open('{:}/template_args.txt'.format(image_dir), 'w') as f:
        json.dump(args_dict, f, indent=2)

    # load initial image (after affine registration)
    image_vol = ants.image_read(initial_image)
    label_vol = ants.image_read(label_image)
    cortical_label_vol = ants.image_read(cortical_label_image)

    padding = (initial_image.split('_')[-1].split('g')[-2].split('.')[0])
    if padding:
        print('image padding by {:}'.format(padding))
        image_vol_shape = image_vol.shape
        padding = int(padding)
        image_vol = ants.pad_image(image_vol, pad_width=(0, padding, 0))
        label_vol = ants.pad_image(label_vol, pad_width=(0, padding, 0))
        cortical_label_vol = ants.pad_image(cortical_label_vol, pad_width=(0, padding, 0))

    # resample in-plane resolution to lower computation
    image_vol = ants.resample_image(image_vol, (template_resolution, 0.5, template_resolution), interp_type=4 ) #bspline
    label_vol = ants.resample_image(label_vol, (template_resolution, 0.5, template_resolution), interp_type=1 ) #NN
    cortical_label_vol = ants.resample_image(cortical_label_vol, (template_resolution, 0.5, template_resolution), interp_type=1 ) #NN

    # sort out strange padding issue
    if padding:
        if np.sum(image_vol[:, (padding // 2)-1, :]) > 0:
             print('fix padding')
             image_vol = ants.pad_image(image_vol, pad_width = ([0, 0], [padding // 2, -padding // 2], [0, 0]))
             label_vol = ants.pad_image(label_vol, pad_width = ([0, 0], [padding // 2, -padding // 2], [0, 0]))
             cortical_label_vol = ants.pad_image(cortical_label_vol, pad_width = ([0, 0], [padding // 2, -padding // 2], [0, 0]))

    print('augmenting image volumes...')
    # to adjust for different resolution in and through plane
    axis_ratio = round(image_vol.spacing[1] / image_vol.spacing[0])
    for v in np.arange(num_versions):
        a_vol, a_labels, a_cortical_labels = augment_image_slices(image_vol, label_vol, cortical_label_vol, direction='coronal', slices=slice_drop)
        s_vol, s_labels, s_cortical_labels = augment_image_slices(a_vol, a_labels, a_cortical_labels, direction='axial', slices=slice_drop*axis_ratio) # axis ratio ensures equiv. amount of distortion along each axis
        aug_vol, aug_labels, aug_cortical_labels = augment_image_slices(s_vol, s_labels, s_cortical_labels, direction='sagittal', slices=slice_drop*axis_ratio)
        # save augmented image data
        ants.image_write(aug_vol, stack_dir + '/random_slices{:03d}_res{:}_drop{:}.nii.gz'.format(v, template_resolution, slice_drop))
        ants.image_write(aug_labels, stack_dir + '/random_slices_labels{:03d}_res{:}_drop{:}.nii.gz'.format(v, template_resolution, slice_drop))
        ants.image_write(aug_cortical_labels, stack_dir + '/random_slices_cortical_labels{:03d}_res{:}_drop{:}.nii.gz'.format(v, template_resolution, slice_drop))
        print('done with {:}/random_slices{:03d}_res{:}_drop{:}.nii.gz'.format(stack_dir, v, template_resolution, slice_drop))

    print('')
    print('aligning slices...')
    augmented_volumes = sorted(glob.glob(stack_dir + '/random_slices???_res{:}_drop{:}.nii.gz'.format(template_resolution, slice_drop)))
    augmented_labels = sorted(glob.glob(stack_dir + '/random_slices_labels???_res{:}_drop{:}.nii.gz'.format(template_resolution, slice_drop)))
    augmented_cortical_labels = sorted(glob.glob(stack_dir + '/random_slices_cortical_labels???_res{:}_drop{:}.nii.gz'.format(template_resolution, slice_drop)))

    for v, volume in tqdm(enumerate(augmented_volumes), total=len(augmented_volumes)):
        # load augmented volume
        aligned_vol = ants.image_read(volume)
        aligned_label_vol = ants.image_read(augmented_labels[v])
        aligned_cortical_label_vol = ants.image_read(augmented_cortical_labels[v])

        # number of coronal slices
        num_sections = aligned_vol.shape[1]

        # space for new volume
        new_vol = np.zeros_like(aligned_vol.numpy())
        new_label_vol = np.zeros_like(aligned_label_vol.numpy())
        new_cortical_label_vol = np.zeros_like(aligned_cortical_label_vol.numpy())

        # space for iterative transforms (one forward, one backward per iteration)
        transform_array = [[[None] for i in range(reg_iter*2)] for j in range(num_sections)]

        # get initial parameters
        init_vol = aligned_vol.copy()
        slice_spacing = (aligned_vol.spacing[0], aligned_vol.spacing[2])

        # begin iterations
        for it in np.arange(reg_iter):
            print('iteration {:}'.format(it+1))
            # FORWARD PASS
            for section in tqdm(np.arange(num_sections), desc='forward pass'):
                # moving section is current section
                moving_slice = ants.from_numpy(init_vol[:,section,:], spacing=slice_spacing).iMath('Normalize')
                # get following sections (as long as they are not empty)
                target_list = [ants.from_numpy(init_vol[:,s,:], spacing=slice_spacing).iMath('Normalize') for s in np.arange(section+1, section+slice_neighbours+1) if s<num_sections]
                target_list = [s for s in target_list if s.sum()>0]

                # original section to be transformed
                orig_slice = ants.from_numpy(aligned_vol[:,section,:], spacing=slice_spacing)

                # warp moving section halfway
                _, tx = halfway_transformation(target_list, moving_slice, tempdir=tmpdir, weighting=slice_weights)

                # store slice deformation
                if tx:
                    transform_array[section][it*2] = [tx]

                # slice to vol
                # compose transforms from this and previous iterations and apply to original slice
                transform_list = list(filter(None, itertools.chain.from_iterable(transform_array[section][:(it*2)+1])))
                new_section = ants.apply_transforms(orig_slice, orig_slice, transform_list)
                new_vol[:,section,:] = new_section.numpy()

            # make new volume
            init_vol = aligned_vol.new_image_like(new_vol)

            # BACKWARDS PASS
            new_vol = np.zeros_like(aligned_vol.numpy())
            print('backwards')
            for section in tqdm(np.arange(num_sections), desc='backwards pass'):
                # moving section is current section
                moving_slice = ants.from_numpy(init_vol[:,section,:], spacing=slice_spacing)
                # get following section
                target_list = [ants.from_numpy(init_vol[:,s,:], spacing=slice_spacing) for s in np.arange(section-slice_neighbours, section)[::-1] if s>-1]
                target_list = [s for s in target_list if s.sum()>0]

                # original section to be transformed
                orig_slice = ants.from_numpy(aligned_vol[:,section,:], spacing=slice_spacing)

                # run reg
                _, tx = halfway_transformation(target_list, moving_slice, tempdir=tmpdir, weighting=slice_weights)

                # store location of slice deformation
                if tx:
                    transform_array[section][(it*2)+1] = [tx]

                # slice to vol
                # compose transforms from this and previous iterations and apply to original slice
                transform_list = list(filter(None, itertools.chain.from_iterable(transform_array[section][:(it*2)+2])))
                new_section = ants.apply_transforms(orig_slice, orig_slice, transform_list)
                new_vol[:,section,:] = new_section.numpy()

                # *apply to label vol?* #
                if it == reg_iter-1:
                    label_slice = ants.from_numpy(aligned_label_vol[:,section,:], spacing=slice_spacing)
                    new_label_section = ants.apply_transforms(label_slice, label_slice, transform_list, interpolator='genericLabel')
                    new_label_vol[:,section,:] = new_label_section.numpy()

                    cortical_label_slice = ants.from_numpy(aligned_cortical_label_vol[:,section,:], spacing=slice_spacing)
                    new_cortical_label_section = ants.apply_transforms(cortical_label_slice, cortical_label_slice, transform_list, interpolator='genericLabel')
                    new_cortical_label_vol[:,section,:] = new_cortical_label_section.numpy()

            # make new volume
            init_vol = aligned_vol.new_image_like(new_vol)


        print('finished')
        print('see: {:}/nonlinear_slice_alignment{:03d}_res{:}_drop{:}.nii.gz.nii.gz'.format(stack_dir, v, template_resolution, slice_drop))
        ants.image_write(init_vol, '{:}/nonlinear_slice_alignment{:03d}_res{:}_drop{:}.nii.gz'.format(stack_dir, v, template_resolution, slice_drop))
        # label image
        final_label_vol = aligned_label_vol.new_image_like(new_label_vol)
        ants.image_write(final_label_vol, '{:}/nonlinear_slice_label_alignment{:03d}_res{:}_drop{:}.nii.gz'.format(stack_dir, v, template_resolution, slice_drop))
        # cortical_label image
        final_cortical_label_vol = aligned_cortical_label_vol.new_image_like(new_cortical_label_vol)
        ants.image_write(final_cortical_label_vol, '{:}/nonlinear_slice_cortical_label_alignment{:03d}_res{:}_drop{:}.nii.gz'.format(stack_dir, v, template_resolution, slice_drop))

        # delete temporary files
        os.system('rm {:}/tmp*nii.gz'.format(tmpdir))

    print('')
    print('building template from augmented volumes...')
    images = [ants.image_read(im).iMath('Normalize') for im in sorted(glob.glob('{:}/nonlinear_slice_alignment*_res{:}_drop{:}.nii.gz'.format(stack_dir, template_resolution, slice_drop)))]

    # create prior image as average of augmented volumes
    im = images[0]
    for i in images[1:]:
        im = im+i
    im = im / len(images)

    prior = ants.resample_image(im, (template_resolution, template_resolution, template_resolution))
    mask = ants.get_mask(prior)

    # build template
    print('building template')
    template = ants.build_template(initial_template=prior, image_list=images, iterations=3, syn_sampling=4, syn_metric='CC', reg_iterations = (100, 100, 50, 50, 10), mask=mask, gradient_step=0.25, verbose=True)
    ants.image_write(template, '{:}/nonlinear_template_res{:}.nii.gz'.format(image_dir, template_resolution))
    print('see: {:}/nonlinear_template_res{:}.nii.gz'.format(image_dir, template_resolution))

    print('')
    print('calculating transforms to template...')
    images = [ants.image_read(im).iMath('Normalize') for im in sorted(glob.glob('AlignedImages/image_stacks/nonlinear_slice_alignment*.nii.gz'))]
    labels = [ants.image_read(im) for im in sorted(glob.glob('AlignedImages/image_stacks/nonlinear_slice_label_alignment*.nii.gz'))]
    cortical_labels = [ants.image_read(im) for im in sorted(glob.glob('AlignedImages/image_stacks/nonlinear_slice_cortical_label_alignment*.nii.gz'))]

    mask = ants.get_mask(template)

    # build template
    for n,i in enumerate(images):
        reg = ants.registration(template, i, type_of_transform='SyN', syn_metric='CC', syn_sampling=4, reg_iterations=(100, 100, 50, 10, 0), mask=mask, grad_step=0.25, verbose=True)
        transformed_labels = ants.apply_transforms(template, labels[n], reg['fwdtransforms'], interpolator='genericLabel')
        transformed_cortical_labels = ants.apply_transforms(template, cortical_labels[n], reg['fwdtransforms'], interpolator='genericLabel')

        ants.image_write(reg['warpedmovout'], 'AlignedImages/image_stacks/nonlinear_to_template_{:03d}.nii.gz'.format(n))
        ants.image_write(transformed_labels, 'AlignedImages/image_stacks/nonlinear_labels_to_template_{:03d}.nii.gz'.format(n))
        ants.image_write(transformed_cortical_labels, 'AlignedImages/image_stacks/nonlinear_cortical_labels_to_template_{:03d}.nii.gz'.format(n))

    print('FINISHED')

def augment_image_slices(image, labels, cortical_labels, direction='coronal', slices=10):
    """
    create a synthetic image by randomly dropping and repeating image slices in stack along
    give direction.

    image: ANTsImage
    labels: ANTsImage
    cortical_labels: ANTsImage
    direction: 'coronal', 'axial' or 'sagittal'
    num_slices: how many slices to drop/repeat
    """
    assert direction in ['coronal', 'axial', 'sagittal'], 'incorrect direction specified'
    if direction == 'coronal':
        _, num_sections, _ = image.shape
    elif direction == 'sagittal':
        num_sections, _, _ = image.shape
    elif direction == 'axial':
        _, _, num_sections = image.shape

    # space for new image
    new_image = np.zeros_like(image.numpy())
    new_label_image = np.zeros_like(labels.numpy())
    new_cortical_label_image = np.zeros_like(cortical_labels.numpy())

    # slices to keep
    keep_slices = np.random.choice(np.arange(num_sections), size=num_sections-slices, replace=False)

    # slices to repeat
    repeat_slices = np.ones(num_sections - slices)
    ridx = np.random.choice(np.arange(num_sections - slices), size=slices, replace=False)
    repeat_slices[ridx] = 2

    # slice indices
    all_slices = np.repeat(np.array(sorted(keep_slices)), repeat_slices.astype(int))

    # replace original values with new slice values
    if direction == 'coronal':
        for n, s in enumerate(all_slices):
            new_image[:,n,:] = image.numpy()[:,s,:]
            new_label_image[:,n,:] = labels.numpy()[:,s,:]
            new_cortical_label_image[:,n,:] = cortical_labels.numpy()[:,s,:]

    elif direction == 'sagittal':
        for n, s in enumerate(all_slices):
            new_image[n,:,:] = image.numpy()[s,:,:]
            new_label_image[n,:,:] = labels.numpy()[s,:,:]
            new_cortical_label_image[n,:,:] = cortical_labels.numpy()[s,:,:]

    elif direction == 'axial':
        for n, s in enumerate(all_slices):
            new_image[:,:,n] = image.numpy()[:,:,s]
            new_label_image[:,:,n] = labels.numpy()[:,:,s]
            new_cortical_label_image[:,:,n] = cortical_labels.numpy()[:,:,s]

    # make new ANTsImage
    new_image = image.new_image_like(new_image)
    new_label_image = labels.new_image_like(new_label_image)
    new_cortical_label_image = labels.new_image_like(new_cortical_label_image)

    return new_image, new_label_image, new_cortical_label_image



def halfway_transformation(fixed_list, moving, tempdir='/tmp', weighting='exp'):
    """
    perform deformable registration between moving and fixed images, calculate
    the halfway transform and apply to the moving section

    fixed_list: list of ANTsImages to act as targets
    moving: ANTsImages, moving image for registration
    tempdir: where to save temporary reg files
    weighting: 'exp, 'linear', None, weighting to apply to target slices

    returns:
    warped: moving image warped towards fixed image
    tx: path_to_transform
    """
    # number of targets
    length = len(fixed_list)
    # return if no targets or empty moving image
    if length == 0 or moving.sum() == 0:
        warped = moving.copy()
        txout = None

        return warped, txout

    # weights for targets
    if weighting == 'linear':
        weight_list = (np.arange(length+1)[::-1] / ((length+1)-1))[:-1]
    elif weighting == 'exp':
        weight_list = np.exp(np.arange(length)[::-1]) / np.exp(length-1)
    else:
        weight_list = np.ones(length)

    mv_extras = []
    for im in np.arange(1,len(fixed_list)):
        if fixed_list[im].sum() > 0:
            mv_extras.append(['CC', fixed_list[im], moving, weight_list[im], 4])

    # registration
    deformation = ants.registration(fixed_list[0], moving, type_of_transform='SyNOnly', \
                                        syn_sampling=4, syn_metric='CC', grad_step=.25, \
                                        flow_sigma=3.0, total_sigma=1.0, \
                                        reg_iterations=[100,100,100,50,50,10], \
                                        verbose=False, write_composite_transform=True, multivariate_extras=mv_extras)

    # load composite transform and ....divide by two
    txout = mktemp(suffix='', dir=tempdir)
    # convert hdf5 to warp image format
    txout = ants.apply_transforms(fixed_list[0], moving, deformation['fwdtransforms'], compose=txout)

    # find and save halfway deformation
    half_tx = ants.image_read(txout) / 2
    ants.image_write(half_tx, txout)

    # apply to moving image
    warped = ants.apply_transforms(fixed_list[0], moving, txout)

    return warped, txout


if __name__ == '__main__':
    import argparse, os

    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise FileNotFoundError(string)

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    def limited_float(x):
        x = float(x)
        if x < 0.05 or x > 0.5:
            raise argparse.ArgumentTypeError(x)
        return x

    parser = argparse.ArgumentParser(description='Perform data augmentation and template construction of aligned sections')

    parser.add_argument('-i', '--initial_image', type=file_path, required=True,
            help='initial volume of affinely aligned slices (.nii.gz)')
    parser.add_argument('-l', '--label_image', type=file_path, required=True,
            help='initial volume of affinely aligned label slices (.nii.gz)')
    parser.add_argument('-c', '--cortical_label_image', type=file_path, required=True,
            help='initial volume of affinely aligned cortical label slices (.nii.gz)')
    parser.add_argument('-s', '--slice_neighbours', type=int, required=True,
            help='number of neighbouring slices used in slice-to-slice nonlinear registrations')
    parser.add_argument('-w', '--slice_weights', choices=['linear', 'exp'], required=True,
                help='neighbouring slice weighting for nonlinear registrations')
    parser.add_argument('-r', '--reg_iter', type=int, required=True,
                help='number of iterations for initial registrations')
    parser.add_argument('-t', '--template_resolution', type=limited_float, required=True, metavar=[0.05-0.5],
            help='number of neighbouring slices used in slice-to-slice nonlinear registrations')
    parser.add_argument('-v', '--num_versions', type=int, required=False, default=50,
                help='number of augmented image versions')
    parser.add_argument('-a', '--slice_drop', type=int, required=False, default=10,
                help='number of slices to drop/repeat in augmentation')
    parser.add_argument('-d', '--tmpdir', type=dir_path, required=False, default='/tmp',
                help='alternative location for temporary registration files (cleared after each iteration)')
    args = parser.parse_args()

    main(args.initial_image, args.label_image, args.cortical_label_image,
        num_versions=args.num_versions,
        slice_drop=args.slice_drop,
        slice_neighbours=args.slice_neighbours,
        slice_weights=args.slice_weights,
        reg_iter=args.reg_iter,
        tmpdir=args.tmpdir,
        template_resolution=args.template_resolution)
