#! /bin/bash

# run freesurfer surface extraction on uBrain volume
mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 atlas/temp/rh.inflated
mris_sphere atlas/temp/rh.inflated atlas/temp/rh.sphere

# create subject directory structure
export SUBJECTS_DIR='./atlas/'
export SUBJ='fs_atlas'
echo 'Creating a subject directory for '${SUBJ} 'in:'
echo ${SUBJECTS_DIR}/${SUBJ}
rm -r ${SUBJECTS_DIR}/${SUBJ}
mksubjdirs ${SUBJECTS_DIR}/${SUBJ}

# move over files to FS directories
cp atlas/temp/rh.inflated ${SUBJECTS_DIR}/${SUBJ}/surf/
cp atlas/temp/rh.smoothwm ${SUBJECTS_DIR}/${SUBJ}/surf/
cp atlas/temp/rh.orig ${SUBJECTS_DIR}/${SUBJ}/surf/
cp atlas/temp/rh.qsphere ${SUBJECTS_DIR}/${SUBJ}/surf/

cp atlas/mgz/T1.mgz ${SUBJECTS_DIR}/${SUBJ}/mri/T1.mgz
cp atlas/mgz/filled.mgz ${SUBJECTS_DIR}/${SUBJ}/mri/filled.mgz
cp atlas/mgz/wm.mgz ${SUBJECTS_DIR}/${SUBJ}/mri/wm.mgz
cp atlas/mgz/seg-vol.mgz ${SUBJECTS_DIR}/${SUBJ}/mri/brain.finalsurfs.mgz

# make surfaces with pseudoT2
mri_fwhm --i ${SUBJECTS_DIR}/${SUBJ}/mri/brain.finalsurfs.mgz --o ${SUBJECTS_DIR}/${SUBJ}/mri/brain.finalsurfs.smo.mgz --smooth-only --fwhm 5
mris_make_surfaces -noaseg -noaparc -T1 brain.finalsurfs.smo -orig_wm orig ${SUBJ} rh
mris_sphere ${SUBJECTS_DIR}/${SUBJ}/surf/rh.inflated ${SUBJECTS_DIR}/${SUBJ}/surf/rh.sphere

# regenerate pial
mris_make_surfaces  -noaseg -noaparc -orig_white white -orig_pial white -nowhite -mgz -T1 brain.finalsurfs.smo ${SUBJ} rh
# midthickness
mris_expand -thickness ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white 0.5 ${SUBJECTS_DIR}/${SUBJ}/surf/rh.mid

# fix up
mris_euler_number ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white
mris_remove_intersection ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white
mris_smooth -n 50 ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.white

mris_euler_number ${SUBJECTS_DIR}/${SUBJ}/surf/rh.pial
mris_remove_intersection ${SUBJECTS_DIR}/${SUBJ}/surf/rh.pial ${SUBJECTS_DIR}/${SUBJ}/surf/rh.pial
mris_smooth -n 50 -nw ${SUBJECTS_DIR}/${SUBJ}/surf/rh.pial ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.pial

mris_euler_number ${SUBJECTS_DIR}/${SUBJ}/surf/rh.mid
mris_remove_intersection ${SUBJECTS_DIR}/${SUBJ}/surf/rh.mid ${SUBJECTS_DIR}/${SUBJ}/surf/rh.mid
mris_smooth -n 50 -nw ${SUBJECTS_DIR}/${SUBJ}/surf/rh.mid ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.mid

mris_inflate ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.white ${SUBJECTS_DIR}/${SUBJ}/surf/rh.inflated
mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 ${SUBJECTS_DIR}/${SUBJ}/surf/rh.inflated
mris_sphere ${SUBJECTS_DIR}/${SUBJ}/surf/rh.inflated ${SUBJECTS_DIR}/${SUBJ}/surf/rh.sphere

# scale to original size and transform to gifti
# get c_ras from original volume
C_R=`mri_info nonlinear_template_res0.15.nii.gz  | grep c_r | awk '{print $NF}'`
C_A=`mri_info nonlinear_template_res0.15.nii.gz  | grep c_a | awk '{print $NF}'`
C_S=`mri_info nonlinear_template_res0.15.nii.gz  | grep c_s | awk '{print $NF}'`

# same from surface
mris_info ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smoothwm >& ${SUBJECTS_DIR}/${SUBJ}/surf/rh_info.txt
sC_R=`cat ${SUBJECTS_DIR}/${SUBJ}/surf/rh_info.txt | grep c_ | awk '{print $3}' | sed 's/(// ; s/,// ; s/)//'`
sC_A=`cat ${SUBJECTS_DIR}/${SUBJ}/surf/rh_info.txt | grep c_ | awk '{print $4}' | sed 's/(// ; s/,// ; s/)//'`
sC_S=`cat ${SUBJECTS_DIR}/${SUBJ}/surf/rh_info.txt | grep c_ | awk '{print $5}' | sed 's/(// ; s/,// ; s/)//'`

# scale surface c_ras
sC_R=$(echo $sC_R*.15 | bc)
sC_A=$(echo $sC_A*.15 | bc)
sC_S=$(echo $sC_S*.15 | bc)

# get difference
dC_R=$(echo $C_R- $sC_R | bc)
dC_A=$(echo $C_A- $sC_A | bc)
dC_S=$(echo $C_S- $sC_S | bc)

echo "0.15 0 0 $dC_R" > ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat
echo "0 0.15 0 $dC_A" >> ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat
echo "0 0 0.15 $dC_S" >> ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat
echo "0 0 0 1" >> ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat

# convert FS surfaces to gifti, scale and centre on original volume
mkdir -p ${SUBJECTS_DIR}/${SUBJ}/surf/gifti
mris_convert --to-scanner ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.white ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.white.surf.gii
wb_command -surface-apply-affine ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.white.surf.gii ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.white.surf.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.white.surf.gii CORTEX_RIGHT  -surface-type ANATOMICAL -surface-secondary-type GRAY_WHITE

mris_convert --to-scanner ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.mid ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.mid.surf.gii
wb_command -surface-apply-affine ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.mid.surf.gii ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.mid.surf.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.mid.surf.gii CORTEX_RIGHT -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS

mris_convert --to-scanner ${SUBJECTS_DIR}/${SUBJ}/surf/rh.smooth.pial ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.pial.surf.gii
wb_command -surface-apply-affine ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.pial.surf.gii ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.pial.surf.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.pial.surf.gii CORTEX_RIGHT -surface-type ANATOMICAL -surface-secondary-type PIAL

mris_convert --to-scanner ${SUBJECTS_DIR}/${SUBJ}/surf/rh.inflated ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.inflated.surf.gii
wb_command -surface-apply-affine ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.inflated.surf.gii ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.inflated.surf.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.inflated.surf.gii CORTEX_RIGHT -surface-type INFLATED

mris_convert --to-scanner ${SUBJECTS_DIR}/${SUBJ}/surf/rh.sphere ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.sphere.surf.gii
wb_command -surface-apply-affine ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.sphere.surf.gii ${SUBJECTS_DIR}/${SUBJ}/surf/affine.mat ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.sphere.surf.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.sphere.surf.gii CORTEX_RIGHT -surface-type SPHERICAL

mris_convert -c rh.curv ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.curv.shape.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.curv.shape.gii CORTEX_RIGHT

mris_convert -c rh.sulc ${SUBJECTS_DIR}/${SUBJ}/surf/rh.white ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.sulc.shape.gii
wb_command -set-structure ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/R.sulc.shape.gii CORTEX_RIGHT

# copy original volume to gifti directory
cp nonlinear_template_res0.15.nii.gz  ${SUBJECTS_DIR}/${SUBJ}/surf/gifti/T1.nii.gz

echo ""
echo "surface extraction complete"
