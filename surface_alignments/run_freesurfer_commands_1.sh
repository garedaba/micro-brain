#! /bin/bash
# run freesurfer surface extraction on fetal brain atlas
# following protocols from here:
# https://prime-re.github.io/structural/surfaces_and_flatmaps_notebook/Surfaces_and_Flatmaps.html
# and here:
# https://freesurfer.net/fswiki/ExVivo

# create dummy volume first and edit voxel dimensions to 1mm isotropic

# step 1: set up and edit nifti headers
# make directories
mkdir -p atlas/mgz atlas/src atlas/temp atlas/src/org

# copy files over: uBrain volume, wm mask from brain tissue labels
cp nonlinear_template_res0.15.nii.gz atlas/src/T1.nii.gz
fslmaths nonlinear_template_res0.15.nii.gz -thr 0.2 atlas/src/brain.nii.gz
cp wm.nii.gz atlas/src/wm.nii.gz

# backup
cp atlas/src/*nii.gz atlas/src/org/

# create dummy volumes with 1mm voxels using dummy volume as a target
wb_command -volume-set-space atlas/src/T1.nii.gz atlas/src/T1.nii.gz -file nonlinear_template_res0.15.VOXEL1.nii.gz
wb_command -volume-set-space atlas/src/brain.nii.gz atlas/src/brain.nii.gz -file nonlinear_template_res0.15.VOXEL1.nii.gz
wb_command -volume-set-space atlas/src/wm.nii.gz atlas/src/wm.nii.gz -file nonlinear_template_res0.15.VOXEL1.nii.gz

# convert to mgz format and downsample
mri_convert -c -sc 255 atlas/src/T1.nii.gz atlas/mgz/T1.mgz
mri_convert -ds 2 2 2  atlas/mgz/T1.mgz atlas/mgz/T1.mgz

mri_convert -c -sc 255 atlas/src/brain.nii.gz atlas/mgz/brain.mgz
mri_convert -ds 2 2 2  atlas/mgz/brain.mgz atlas/mgz/brain.mgz

mri_convert -c -rt nearest atlas/src/wm.nii.gz atlas/mgz/wm.mgz
mri_convert -ds 2 2 2  -rt nearest atlas/mgz/wm.mgz atlas/mgz/wm.mgz

# if brain mask edits are needed - copy edited version to atlas/mgz/brainmask.mgz
if [ -f brainmask_edited.mgz ]; then
  echo "edited brain mask found"
  cp brainmask_edited.mgz atlas/mgz/brain.mgz
fi
# if wm mask edits are needed - copy edited version to atlas/mgz/wm.mgz
if [ -f wm_edited.mgz ]; then
  echo "edited wm mask found"
  cp wm_edited.mgz atlas/mgz/wm.mgz
fi

# create pseudovolume for surface extraction
mri_convert atlas/mgz/brain.mgz atlas/mgz/brain-mask.nii.gz
fslmaths atlas/mgz/brain-mask.nii.gz -thr 10 -bin atlas/mgz/brain-mask.nii.gz

mri_convert atlas/mgz/wm.mgz atlas/mgz/wm-mask.nii.gz
fslmaths atlas/mgz/wm-mask.nii.gz -thr 10 -bin atlas/mgz/wm-mask.nii.gz

fslmaths atlas/mgz/brain-mask.nii.gz -add atlas/mgz/wm-mask.nii.gz -bin -sub atlas/mgz/wm-mask.nii.gz -mul 80 atlas/mgz/brain-mask.nii.gz
fslmaths atlas/mgz/wm-mask -mul 110 -add atlas/mgz/brain-mask atlas/mgz/seg-vol.nii.gz

mri_convert -odt uchar -ns 1 atlas/mgz/seg-vol.nii.gz atlas/mgz/seg-vol.mgz

rm atlas/mgz/wm-mask.nii.gz atlas/mgz/brain-mask.nii.gz atlas/mgz/seg-vol.nii.gz

# create 'filled' wm mask for only one hemisphere
mri_extract_largest_CC -T 127 -hemi rh atlas/mgz/wm.mgz atlas/mgz/filled.mgz
cp atlas/mgz/wm.mgz atlas/mgz/wm_nofix.mgz

# create initial surface
# right hemisphere
mri_pretess atlas/mgz/filled.mgz 127 atlas/mgz/brain.mgz atlas/mgz/wm_filled-pretess127.mgz
mri_tessellate atlas/mgz/wm_filled-pretess127.mgz 127 atlas/temp/rh.orig.nofix

cp atlas/temp/rh.orig.nofix atlas/temp/rh.orig

# post-process tesselation
mris_extract_main_component atlas/temp/rh.orig.nofix atlas/temp/rh.orig.nofix
mris_smooth -nw -n 50 atlas/temp/rh.orig.nofix atlas/temp/rh.smoothwm.nofix
mris_inflate atlas/temp/rh.smoothwm.nofix atlas/temp/rh.inflated.nofix
mris_sphere -q atlas/temp/rh.inflated.nofix atlas/temp/rh.qsphere.nofix
cp atlas/temp/rh.inflated.nofix atlas/temp/rh.inflated

# fix topology
mris_euler_number atlas/temp/rh.orig
mris_remove_intersection atlas/temp/rh.orig atlas/temp/rh.orig
mris_smooth -nw -n 50 atlas/temp/rh.orig atlas/temp/rh.smoothwm
mris_inflate atlas/temp/rh.smoothwm atlas/temp/rh.inflated
mris_sphere -q atlas/temp/rh.inflated atlas/temp/rh.qsphere

echo ""
echo "initial surface extration complete"
echo "inspect surfaces and edit as needed:"
echo "freeview -v atlas/mgz/brain.mgz -v atlas/mgz/wm.mgz -f atlas/temp/rh.smoothwm atlas/temp/rh.inflated"
