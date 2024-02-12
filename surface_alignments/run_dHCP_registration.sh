#! /bin/bash

# using manually-defined ROI, align ubrain cortical surface to dHCP fetal template

#1. ensure A1 cortex is not labeled in superiortemporal label
wb_command -metric-math '(y - x) > 0' cortical_roi/ubrain.superiortemporal.func.gii -var y cortical_roi/ubrain.superiortemporal.func.gii -var x cortical_roi/ubrain.a1.func.gii
wb_command -metric-math '(y - x) > 0' cortical_roi/fetal.superiortemporal.func.gii -var y cortical_roi/fetal.superiortemporal.func.gii -var x cortical_roi/fetal.a1.func.gii

#2. create smoothed gradient images used to drive registration
cd cortical_roi

# cortical mask
wb_command -metric-resample mask-fetal.cortex.func.gii ../fetal_atlas/transforms/fetal.week36_to_week21.right.sphere.reg.surf.gii ../fetal_atlas/fetal.week21.right.sphere.surf.gii ADAP_BARY_AREA week21-mask-fetal.cortex.func.gii  -area-surfs ../fetal_atlas/fetal.week36.right.midthickness.surf.gii ../fetal_atlas/fetal.week21.right.midthickness.surf.gii -largest

for i in fetal*func*; do
  wb_command -metric-resample $i ../fetal_atlas/transforms/fetal.week36_to_week21.right.sphere.reg.surf.gii ../fetal_atlas/fetal.week21.right.sphere.surf.gii ADAP_BARY_AREA week21-${i}  -area-surfs ../fetal_atlas/fetal.week36.right.midthickness.surf.gii ../fetal_atlas/fetal.week21.right.midthickness.surf.gii -largest
  wb_command -metric-smoothing ../fetal_atlas/fetal.week21.right.midthickness.surf.gii week21-${i} 2 gradient-week21-${i}
done

for i in ubrain*func*; do
  wb_command -metric-smoothing ../ubrain/R.mid.surf.gii ${i} 2 gradient-${i}
done

cd -

#3. merge all gradient metrics into a single gifti
wb_command -metric-merge ubrain.allroi.shape.gii \
          -metric cortical_roi/gradient-ubrain.a1.func.gii \
          -metric cortical_roi/gradient-ubrain.cingulate.func.gii \
          -metric cortical_roi/gradient-ubrain.frontal.func.gii \
          -metric cortical_roi/gradient-ubrain.insula.func.gii \
          -metric cortical_roi/gradient-ubrain.m1.func.gii \
          -metric cortical_roi/gradient-ubrain.midline.func.gii \
          -metric cortical_roi/gradient-ubrain.occipital.func.gii \
          -metric cortical_roi/gradient-ubrain.parahippocampal.func.gii \
          -metric cortical_roi/gradient-ubrain.parietal.func.gii \
          -metric cortical_roi/gradient-ubrain.s1.func.gii \
          -metric cortical_roi/gradient-ubrain.superiortemporal.func.gii \
          -metric cortical_roi/gradient-ubrain.ventrolateraltemporal.func.gii

#4. as above but for fetal_atlas
wb_command -metric-merge fetal.allroi.shape.gii \
            -metric cortical_roi/gradient-week21-fetal.a1.func.gii \
            -metric cortical_roi/gradient-week21-fetal.cingulate.func.gii \
            -metric cortical_roi/gradient-week21-fetal.frontal.func.gii \
            -metric cortical_roi/gradient-week21-fetal.insula.func.gii \
            -metric cortical_roi/gradient-week21-fetal.m1.func.gii \
            -metric cortical_roi/gradient-week21-fetal.midline.func.gii \
            -metric cortical_roi/gradient-week21-fetal.occipital.func.gii \
            -metric cortical_roi/gradient-week21-fetal.parahippocampal.func.gii \
            -metric cortical_roi/gradient-week21-fetal.parietal.func.gii \
            -metric cortical_roi/gradient-week21-fetal.s1.func.gii \
            -metric cortical_roi/gradient-week21-fetal.superiortemporal.func.gii \
            -metric cortical_roi/gradient-week21-fetal.ventrolateraltemporal.func.gii

#5. surface registration
# alignment with sulcal data first
./msm --conf=configs/config_ubrain_to_fetal \
      --inmesh=ubrain/R.sphere.surf.gii \
      --refmesh=fetal_atlas/fetal.week21.right.sphere.surf.gii \
      --indata=ubrain/R.negsulc.nomidline.shape.gii \
      --refdata=fetal_atlas/fetal.week21.right.sulc.shape.gii  \
      --out=ubrain.sulc_reg.R. \
      --verbose

# multivariate alignment using cortical ROIs
./msm --conf=configs/config_ubrain_to_fetal \
      --inmesh=ubrain/R.sphere.surf.gii \
      --refmesh=fetal_atlas/fetal.week21.right.sphere.surf.gii \
      --trans=ubrain.sulc_reg.R.sphere.reg.surf.gii \
      --indata=ubrain.allroi.shape.gii \
      --refdata=fetal.allroi.shape.gii \
      --out=all_features.R. \
      --verbose

# 6. transform cortical labels onto dHCP cortical surface
wb_command -label-resample ubrain.cortical.label.gii all_features.R.sphere.reg.surf.gii fetal_atlas/fetal.week21.right.sphere.surf.gii  ADAP_BARY_AREA ubrain-labels-to-fetal-template.label.gii -area-surfs ubrain/R.mid.surf.gii fetal_atlas/fetal.week21.right.midthickness.surf.gii
echo "cortical registrations complete"
echo "see: ubrain-labels-to-fetal-template.label.gii"
