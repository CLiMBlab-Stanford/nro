#!/usr/bin/env bash

# Execute the HCP MSMAll calibration route with durable internal checkpoints.
# The runner owns this script as one optional anatomical branch. Each named
# stage remains resumable after worker timeout or preemption.

set -euo pipefail

configuration=${1:?usage: msmall_driver.sh CONFIGURATION}
source "$configuration"
atlas_validator="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/msmall_validate_atlas.py"

set +e
source /opt/qunex/env/qunex_environment.sh >/dev/null 2>&1
setup_status=$?
set -e
if [[ "$setup_status" -ne 0 ]]; then
    echo "QuNex environment setup failed with status $setup_status" >&2
    exit "$setup_status"
fi

export FS_LICENSE="$fs_license"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${NRO_CPUS_PER_TASK:-2}}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$OMP_NUM_THREADS"
export FSLPARALLEL="$OMP_NUM_THREADS"

case "$matlab_run_mode" in
    octave) hcp_matlab_run_mode=2 ;;
    *)
        echo "Unsupported MSMAll MATLAB run mode: $matlab_run_mode" >&2
        exit 2
        ;;
esac
if [[ "$fix_training_model" != HCP_Style_Single_Multirun_Dedrift ]]; then
    echo "Unsupported MSMAll FIX model: $fix_training_model" >&2
    exit 2
fi

if [[ -z "$work_root" || "$work_root" == / ]]; then
    echo "Unsafe MSMAll work root: $work_root" >&2
    exit 2
fi

study="$work_root/study"
session="${subject#sub-}_msmall"
session_root="$study/$session"
markers="$work_root/markers"
logs="$work_root/logs"
runtime="$work_root/runtime"
mkdir -p "$study" "$markers" "$logs" "$runtime"

recorded_structural=$(cat "$work_root/structural.fingerprint" 2>/dev/null || true)
recorded_calibration=$(cat "$work_root/calibration.fingerprint" 2>/dev/null || true)
if [[ -n "$recorded_structural" && "$recorded_structural" != "$structural_fingerprint" ]]; then
    echo "MSMAll structural inputs changed; clearing the private MSMAll branch"
    rm -rf "$study" "$markers" "$runtime"
    rm -f "$work_root/complete"
    mkdir -p "$study" "$markers" "$runtime"
elif [[ -n "$recorded_calibration" && "$recorded_calibration" != "$calibration_fingerprint" ]]; then
    echo "MSMAll calibration changed; preserving completed structural reconstruction"
    rm -f "$markers/inventory.complete" "$markers/postfreesurfer.complete"
    rm -f "$markers"/fmri_*.complete "$markers/multirun_fix.complete"
    rm -f "$markers/prepare_msmall.complete" "$markers/msmall.complete"
    rm -f "$markers/dedrift.complete" "$markers/validate.complete" "$work_root/complete"
    rm -rf "$runtime" "$session_root/MNINonLinear/Native"
    rm -rf "$session_root/MNINonLinear"/fsaverage_LR*k
    rm -rf "$session_root/MNINonLinear/Results" "$session_root"/rfMRI_REST*
    mkdir -p "$runtime"
fi
printf '%s\n' "$structural_fingerprint" > "$work_root/structural.fingerprint"
printf '%s\n' "$calibration_fingerprint" > "$work_root/calibration.fingerprint"

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Missing required MSMAll input: $1" >&2
        exit 2
    fi
}

run_stage() {
    local name=$1
    shift
    local marker="$markers/$name.complete"
    if [[ -s "$marker" ]]; then
        echo "Reuse completed MSMAll stage: $name"
        return
    fi
    echo "Start MSMAll stage: $name"
    if "$@" > "$logs/$name.log" 2>&1; then
        :
    else
        local status=$?
        echo "MSMAll stage failed: $name (last 200 log lines follow)" >&2
        tail -n 200 "$logs/$name.log" >&2
        return "$status"
    fi
    printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$marker"
    echo "Completed MSMAll stage: $name"
}

clear_calibration_checkpoints() {
    rm -f "$markers"/fmri_*.complete "$markers/multirun_fix.complete"
    rm -f "$markers/prepare_msmall.complete" "$markers/msmall.complete"
    rm -f "$markers/dedrift.complete" "$markers/validate.complete"
    rm -f "$work_root/complete"
}

clear_after() {
    local stage=$1
    case "$stage" in
        prefreesurfer)
            rm -f "$markers/masked_atlas.complete" "$markers/freesurfer.complete"
            rm -f "$markers/postfreesurfer.complete"
            clear_calibration_checkpoints
            ;;
        masked_atlas)
            rm -f "$markers/freesurfer.complete" "$markers/postfreesurfer.complete"
            clear_calibration_checkpoints
            ;;
        freesurfer)
            rm -f "$markers/postfreesurfer.complete"
            clear_calibration_checkpoints
            ;;
        postfreesurfer)
            clear_calibration_checkpoints
            ;;
        fmri_volume_*)
            rm -f "$markers/fmri_surface_${stage#fmri_volume_}.complete"
            rm -f "$markers/multirun_fix.complete" "$markers/prepare_msmall.complete"
            rm -f "$markers/msmall.complete" "$markers/dedrift.complete"
            rm -f "$markers/validate.complete" "$work_root/complete"
            ;;
        fmri_surface_*|multirun_fix)
            rm -f "$markers/multirun_fix.complete" "$markers/prepare_msmall.complete"
            rm -f "$markers/msmall.complete" "$markers/dedrift.complete"
            rm -f "$markers/validate.complete" "$work_root/complete"
            ;;
        prepare_msmall)
            rm -f "$markers/msmall.complete" "$markers/dedrift.complete"
            rm -f "$markers/validate.complete" "$work_root/complete"
            ;;
        msmall)
            rm -f "$markers/dedrift.complete" "$markers/validate.complete"
            rm -f "$work_root/complete"
            ;;
        dedrift)
            rm -f "$markers/validate.complete" "$work_root/complete"
            ;;
        validate)
            rm -f "$work_root/complete"
            ;;
    esac
}

verify_checkpoint() {
    local name=$1
    shift
    local marker="$markers/$name.complete"
    [[ -s "$marker" ]] || return 0
    local path
    for path in "$@"; do
        if [[ ! -s "$path" ]]; then
            echo "MSMAll checkpoint is incomplete; rebuilding stage: $name"
            rm -f "$marker"
            clear_after "$name"
            return 0
        fi
    done
}

join_at() {
    local IFS=@
    printf '%s' "$*"
}

record_inventory() {
    local output="$work_root/input_manifest.tsv"
    local temporary="$output.tmp"
    {
        printf 'role\tpath\tsha256\n'
        local path index
        for path in "${t1w[@]}"; do
            printf 'T1w\t%s\t' "$path"
            sha256sum "$path" | cut -d' ' -f1
        done
        for path in "${t2w[@]}"; do
            printf 'T2w\t%s\t' "$path"
            sha256sum "$path" | cut -d' ' -f1
        done
        for index in "${!run_names[@]}"; do
            printf '%s\t%s\t' "${run_names[$index]}" "${run_paths[$index]}"
            sha256sum "${run_paths[$index]}" | cut -d' ' -f1
        done
    } > "$temporary"
    mv "$temporary" "$output"
    {
        "$HCPPIPEDIR/show_version" --short || true
        recon-all -version || true
        wb_command -version || true
        printf 'MSM executable SHA256: '
        sha256sum "$(command -v msm)" | cut -d' ' -f1
        /opt/fsl/fsl/bin/python -c 'import pyfix; print(pyfix.__file__)' || true
        /opt/fsl/fsl/bin/python - <<'PY'
import hashlib
from pathlib import Path

import pyfix

root = Path(pyfix.__file__).parent
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(path.read_bytes())
print(f"pyFIX package SHA256: {digest.hexdigest()}")
PY
        printf 'FIX model: %s\n' "$fix_training_model"
        octave --version | sed -n '1p' || true
    } > "$work_root/software_versions.txt" 2>&1
}

run_prefreesurfer() {
    "$HCPPIPEDIR/PreFreeSurfer/PreFreeSurferPipeline.sh" \
        --path="$study" \
        --session="$session" \
        --t1="$(join_at "${t1w[@]}")" \
        --t2="$(join_at "${t2w[@]}")" \
        --t1template="$HCPPIPEDIR_Templates/MNI152_T1_0.7mm.nii.gz" \
        --t1templatebrain="$HCPPIPEDIR_Templates/MNI152_T1_0.7mm_brain.nii.gz" \
        --t1template2mm="$HCPPIPEDIR_Templates/MNI152_T1_2mm.nii.gz" \
        --t2template="$HCPPIPEDIR_Templates/MNI152_T2_0.7mm.nii.gz" \
        --t2templatebrain="$HCPPIPEDIR_Templates/MNI152_T2_0.7mm_brain.nii.gz" \
        --t2template2mm="$HCPPIPEDIR_Templates/MNI152_T2_2mm.nii.gz" \
        --templatemask="$HCPPIPEDIR_Templates/MNI152_T1_0.7mm_brain_mask.nii.gz" \
        --template2mmmask="$HCPPIPEDIR_Templates/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
        --brainsize=150 \
        --fnirtconfig="$HCPPIPEDIR_Config/T1_2_MNI152_2mm.cnf" \
        --fmapmag=NONE --fmapphase=NONE --fmapcombined=NONE --echodiff=NONE \
        --SEPhaseNeg=NONE --SEPhasePos=NONE --seechospacing=NONE \
        --seunwarpdir=NONE --t1samplespacing=NONE --t2samplespacing=NONE \
        --unwarpdir=NONE --gdcoeffs=NONE --avgrdcmethod=NONE --topupconfig=NONE \
        --processing-mode=HCPStyleData
}

repair_atlas_registration() {
    local t1_dir="$session_root/T1w"
    local atlas="$session_root/MNINonLinear"
    local staging="$session_root/.MNINonLinear.masked.tmp"
    local templates="$HCPPIPEDIR/global/templates"
    rm -rf "$staging"
    mkdir -p "$staging/xfms"
    fslmaths "$t1_dir/T1w_acpc_dc_restore_brain.nii.gz" -bin "$staging/T1w_brain_mask"
    flirt -interp spline -dof 12 \
        -in "$t1_dir/T1w_acpc_dc_restore_brain.nii.gz" \
        -ref "$templates/MNI152_T1_0.7mm_brain.nii.gz" \
        -omat "$staging/xfms/acpc2MNILinear.mat" \
        -out "$staging/xfms/T1w_restore_brain_to_MNILinear.nii.gz"
    fnirt \
        --in="$t1_dir/T1w_acpc_dc_restore.nii.gz" \
        --ref="$templates/MNI152_T1_2mm.nii.gz" \
        --aff="$staging/xfms/acpc2MNILinear.mat" \
        --inmask="$staging/T1w_brain_mask.nii.gz" \
        --refmask="$templates/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
        --applyinmask=1 --applyrefmask=1 --impinm=0 --imprefm=0 \
        --fout="$staging/xfms/acpc_dc2standard.nii.gz" \
        --jout="$staging/xfms/NonlinearRegJacobians.nii.gz" \
        --refout="$staging/xfms/IntensityModulatedT1.nii.gz" \
        --iout="$staging/xfms/2mmReg.nii.gz" \
        --logout="$staging/xfms/NonlinearReg.txt" \
        --intout="$staging/xfms/NonlinearIntensities.nii.gz" \
        --cout="$staging/xfms/NonlinearReg.nii.gz" \
        --config="$HCPPIPEDIR_Config/T1_2_MNI152_2mm.cnf"
    invwarp -w "$staging/xfms/acpc_dc2standard.nii.gz" \
        -o "$staging/xfms/standard2acpc_dc.nii.gz" \
        -r "$templates/MNI152_T1_2mm.nii.gz"

    local modality source
    for modality in T1w T2w; do
        source="$t1_dir/${modality}_acpc_dc.nii.gz"
        applywarp --rel --interp=spline -i "$source" \
            -r "$templates/MNI152_T1_0.7mm.nii.gz" \
            -w "$staging/xfms/acpc_dc2standard.nii.gz" \
            -o "$staging/${modality}.nii.gz"
        applywarp --rel --interp=spline \
            -i "$t1_dir/${modality}_acpc_dc_restore.nii.gz" \
            -r "$templates/MNI152_T1_0.7mm.nii.gz" \
            -w "$staging/xfms/acpc_dc2standard.nii.gz" \
            -o "$staging/${modality}_restore.nii.gz"
        fslmaths "$t1_dir/${modality}_acpc_dc_restore_brain.nii.gz" -bin \
            "$staging/${modality}_mask"
        applywarp --rel --interp=nn -i "$staging/${modality}_mask.nii.gz" \
            -r "$templates/MNI152_T1_0.7mm.nii.gz" \
            -w "$staging/xfms/acpc_dc2standard.nii.gz" \
            -o "$staging/${modality}_mask_mni.nii.gz"
        fslmaths "$staging/${modality}_restore.nii.gz" \
            -mas "$staging/${modality}_mask_mni.nii.gz" \
            "$staging/${modality}_restore_brain.nii.gz"
    done
    cp "$staging/T1w.nii.gz" "$staging/T1w_orig.nii.gz"

    /opt/fsl/fsl/bin/python "$atlas_validator" \
        --subject "$staging/T1w_restore_brain.nii.gz" \
        --subject-mask "$staging/T1w_mask_mni.nii.gz" \
        --reference "$templates/MNI152_T1_0.7mm_brain.nii.gz" \
        --reference-mask "$templates/MNI152_T1_0.7mm_brain_mask.nii.gz" \
        --jacobian "$staging/xfms/NonlinearRegJacobians.nii.gz" \
        --output "$staging/registration_qc.json"
    rm -rf "$atlas"
    mv "$staging" "$atlas"
}

run_freesurfer() {
    # A failed recon-all tree is not a trustworthy checkpoint. The enclosing
    # stage marker, rather than FreeSurfer's partial directory, governs reuse.
    rm -rf "$session_root/T1w/$session"
    "$HCPPIPEDIR/FreeSurfer/FreeSurferPipeline.sh" \
        --session="$session" --session-dir="$session_root/T1w" \
        --t1w-image="$session_root/T1w/T1w_acpc_dc_restore.nii.gz" \
        --t1w-brain="$session_root/T1w/T1w_acpc_dc_restore_brain.nii.gz" \
        --t2w-image="$session_root/T1w/T2w_acpc_dc_restore.nii.gz" \
        --seed=1234 --processing-mode=HCPStyleData \
        --extra-reconall-arg=-notal-check
}

run_postfreesurfer() {
    "$HCPPIPEDIR/PostFreeSurfer/PostFreeSurferPipeline.sh" \
        --study-folder="$study" --session="$session" \
        --surfatlasdir="$HCPPIPEDIR_Templates/standard_mesh_atlases" \
        --grayordinatesdir="$HCPPIPEDIR_Templates/91282_Greyordinates" \
        --grayordinatesres="$grayordinates_resolution_mm" \
        --hiresmesh="$high_resolution_mesh" --lowresmesh="$low_resolution_mesh" \
        --subcortgraylabels="$HCPPIPEDIR_Config/FreeSurferSubcorticalLabelTableLut.txt" \
        --freesurferlabels="$HCPPIPEDIR_Config/FreeSurferAllLut.txt" \
        --refmyelinmaps="$HCPPIPEDIR_Templates/standard_mesh_atlases/Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii" \
        --regname="$input_registration" --use-ind-mean=YES \
        --processing-mode=HCPStyleData
}

run_fmri_volume() {
    local index=$1
    "$HCPPIPEDIR/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh" \
        --studyfolder="$study" --session="$session" \
        --fmritcs="${run_paths[$index]}" --fmriname="${run_names[$index]}" \
        --fmrires="$functional_resolution_mm" --biascorrection=SEBASED \
        --fmriscout=NONE --mctype=MCFLIRT --gdcoeffs=NONE \
        --dcmethod=TOPUP_MISMATCHED \
        --echospacing="${run_echo_spacing[$index]}" \
        --unwarpdir="${run_unwarp_direction[$index]}" \
        --SEPhaseNeg="${se_negative[$index]}" --SEPhasePos="${se_positive[$index]}" \
        --topupconfig="$HCPPIPEDIR_Config/b02b0.cnf" \
        --seechospacing="${fieldmap_echo_spacing[$index]}" \
        --seunwarpdir="${run_unwarp_direction[$index]%\-}" \
        --usejacobian=TRUE --processing-mode=HCPStyleData \
        --matlab-run-mode="$hcp_matlab_run_mode" \
        --wb-resample=TRUE
}

run_fmri_surface() {
    local index=$1
    "$HCPPIPEDIR/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh" \
        --studyfolder="$study" --session="$session" \
        --fmriname="${run_names[$index]}" --lowresmesh="$low_resolution_mesh" \
        --fmrires="$functional_resolution_mm" \
        --smoothingFWHM="$surface_smoothing_fwhm_mm" \
        --grayordinatesres="$grayordinates_resolution_mm" \
        --regname="$input_registration" --fmri-qc=YES --goodvoxel=YES
}

run_fix() {
    local results="$session_root/MNINonLinear/Results"
    local concat="$results/rfMRI_REST_CONCAT/rfMRI_REST_CONCAT"
    local ica="${concat}_hp${high_pass_seconds}.ica"
    local fix_driver="$runtime/hcp_fix_multi_run"
    local inputs= names name
    names=$(join_at "${run_names[@]}")
    for name in "${run_names[@]}"; do
        [[ -z "$inputs" ]] || inputs+=@
        inputs+="$results/$name/$name"
    done
    sed \
        -e '/# run fix feature selection/i\
if [[ ! -s "${concatfmrihp}.ica/.fix" ]]; then' \
        -e '/grep -h Noise .*concatfmrihp.*\.ica\/\.fix/a\
else\
    log_Msg "Reusing existing pyFIX classification checkpoint"\
fi' \
        -e 's|${this_script_dir}/scripts|${HCPPIPEDIR}/ICAFIX/scripts|' \
        -e "s| fix_3_clean('| addpath('/opt/HCP/HCPpipelines/ICAFIX/scripts'); rmpath('/opt/HCP/HCPpipelines/global/matlab/icasso122'); fix_3_clean('|" \
        "$HCPPIPEDIR/ICAFIX/hcp_fix_multi_run" > "$fix_driver"
    chmod +x "$fix_driver"
    local -a reuse=()
    if [[ -s "$ica/.fix" && -s "$ica/filtered_func_data.ica/melodic_mix" ]]; then
        reuse+=(--reuse-existing-ica=TRUE)
    fi
    "$fix_driver" \
        --fmri-names="$inputs" --high-pass="$high_pass_seconds" \
        --concat-fmri-name="$concat" --motion-regression=FALSE \
        --enable-legacy-fix=FALSE --fix-threshold="$fix_threshold" \
        --delete-intermediates=FALSE --processing-mode=HCPStyleData \
        --ica-method=MELODIC --parallel-limit="$OMP_NUM_THREADS" \
        --matlab-run-mode="$hcp_matlab_run_mode" "${reuse[@]}"
    printf '%s\n' "$names" > "$work_root/fix_run_names.txt"
}

prepare_msmall() {
    local results="$session_root/MNINonLinear/Results/rfMRI_REST_CONCAT"
    local variance="$results/rfMRI_REST_CONCAT_Atlas_hp${high_pass_seconds}_clean_vn.dscalar.nii"
    local original="${variance%.dscalar.nii}_before_floor.dscalar.nii"
    local temporary="${variance}.tmp"
    require_file "$variance"
    [[ -s "$original" ]] || cp "$variance" "$original"
    "$CARET7DIR/wb_command" -cifti-math 'max(variance, 0.001)' "$temporary" \
        -var variance "$original"
    mv "$temporary" "$variance"
}

prepare_msmall_runtime() {
    mkdir -p "$runtime/bin" "$runtime/matlab"
    cp "$HCPPIPEDIR/MSMAll/scripts/MSMAll.sh" "$runtime/MSMAll.sh"
    sed -i \
        -e 's/mkdir ${NativeFolder}\/${RegName}/mkdir -p ${NativeFolder}\/${RegName}/g' \
        -e '/rm -r "${NativeFolder:?}\/${RegName}"/d' \
        -e "s|mPath=\"\${HCPPIPEDIR}/MSMAll/scripts\"|mPath=\"$runtime/matlab\"|" \
        "$runtime/MSMAll.sh"
    cp "$HCPPIPEDIR/MSMAll/scripts/MSMregression.m" "$runtime/matlab/MSMregression.m"
    sed -i '/SpatialWeightscii=BO;/i\        corrs(~isfinite(corrs))=0;' \
        "$runtime/matlab/MSMregression.m"
    cat > "$runtime/bin/msm" <<'MSM_WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
output_prefix=
declare -a arguments=() metrics=()
for argument in "$@"; do
    case "$argument" in
        --debug) ;;
        --out=*) output_prefix=${argument#--out=}; arguments+=("$argument") ;;
        --indata=*|--inweight=*|--refdata=*|--refweight=*)
            metrics+=("${argument#*=}"); arguments+=("$argument") ;;
        *) arguments+=("$argument") ;;
    esac
done
[[ -n "$output_prefix" ]] || { echo "MSM invocation has no output prefix" >&2; exit 2; }
[[ ! -s "${output_prefix}sphere.reg.surf.gii" ]] || exit 0
for metric in "${metrics[@]}"; do
    statistics=$("$CARET7DIR/wb_command" -metric-stats "$metric" -reduce MEAN)
    if grep -Eqi '(^|[[:space:]])(nan|[-+]?inf)([[:space:]]|$)' <<<"$statistics"; then
        echo "MSM input contains non-finite values: $metric" >&2
        exit 2
    fi
done
"$NRO_REAL_MSM" "${arguments[@]}"
MSM_WRAPPER
    chmod +x "$runtime/MSMAll.sh" "$runtime/bin/msm"
}

run_msmall() {
    prepare_msmall_runtime
    local names templates pipeline
    names=$(join_at "${run_names[@]}")
    templates="$HCPPIPEDIR/global/templates/MSMAll"
    pipeline="$runtime/MSMAllPipeline.sh"
    cp "$HCPPIPEDIR/MSMAll/MSMAllPipeline.sh" "$pipeline"
    sed -i "s|\"\${HCPPIPEDIR}\"/MSMAll/scripts/\"\${ModuleName}\"|$runtime/MSMAll.sh|" "$pipeline"
    chmod +x "$pipeline"
    export NRO_REAL_MSM=/opt/MSM_HOCR_v3/msm
    export MSMBINDIR="$runtime/bin"
    "$pipeline" \
        --study-folder="$study" --session="$session" --fmri-names-list= \
        --multirun-fix-names="$names" \
        --multirun-fix-concat-name=rfMRI_REST_CONCAT \
        --multirun-fix-names-to-use="$names" --output-fmri-name=rfMRI_REST \
        --high-pass="$high_pass_seconds" \
        --fmri-proc-string="_Atlas_hp${high_pass_seconds}_clean" \
        --msm-all-templates="$templates" \
        --myelin-target-file="$templates/Q1-Q6_RelatedParcellation210.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii" \
        --input-registration-name="$input_registration" \
        --output-registration-name="${output_registration}_InitialReg" \
        --high-res-mesh="$high_resolution_mesh" --low-res-mesh="$low_resolution_mesh" \
        --iteration-modes="$iteration_modes" --method="$method" \
        --ica-dim="$ica_dimension" --matlab-run-mode="$hcp_matlab_run_mode"
}

run_dedrift() {
    local names templates
    names=$(join_at "${run_names[@]}")
    templates="$HCPPIPEDIR/global/templates/MSMAll"
    "$HCPPIPEDIR/DeDriftAndResample/DeDriftAndResamplePipeline.sh" \
        --study-folder="$study" --subject="$session" \
        --high-res-mesh="$high_resolution_mesh" --low-res-meshes="$low_resolution_mesh" \
        --registration-name="${output_registration}_InitialReg_2_d${ica_dimension}_${method}" \
        --dedrift-reg-files="$templates/DeDriftingGroup.L.sphere.DeDriftMSMAll.164k_fs_LR.surf.gii@$templates/DeDriftingGroup.R.sphere.DeDriftMSMAll.164k_fs_LR.surf.gii" \
        --concat-reg-name="$output_registration" \
        --maps=sulc@curvature@corrThickness@thickness \
        --myelin-maps=MyelinMap@SmoothedMyelinMap \
        --multirun-fix-concat-names=rfMRI_REST_CONCAT \
        --multirun-fix-names="$names" \
        --multirun-fix-extract-concat-names=rfMRI_REST \
        --multirun-fix-extract-names="$names" --multirun-fix-extract-volume=TRUE \
        --fix-names= --dont-fix-names= \
        --smoothing-fwhm="$surface_smoothing_fwhm_mm" \
        --high-pass="$high_pass_seconds" --matlab-run-mode="$hcp_matlab_run_mode" \
        --motion-regression=FALSE \
        --myelin-target-file="$templates/Q1-Q6_RelatedParcellation210.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"
}

validate_final() {
    local native="$session_root/MNINonLinear/Native"
    local atlas="$session_root/MNINonLinear/fsaverage_LR${low_resolution_mesh}k"
    local hemisphere
    for hemisphere in L R; do
        require_file "$native/$session.$hemisphere.sphere.${output_registration}.native.surf.gii"
        require_file "$native/$session.$hemisphere.sphere.reg.native.surf.gii"
        require_file "$native/$session.$hemisphere.sphere.reg.reg_LR.native.surf.gii"
        require_file "$atlas/$session.$hemisphere.sphere.${low_resolution_mesh}k_fs_LR.surf.gii"
        local surface
        for surface in white midthickness pial inflated; do
            require_file "$atlas/$session.$hemisphere.${surface}_MSMAll.${low_resolution_mesh}k_fs_LR.surf.gii"
        done
    done
}

finalize() {
    validate_final
    printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$work_root/complete"
}

for path in "${t1w[@]}" "${t2w[@]}" "$fs_license"; do require_file "$path"; done
for index in "${!run_names[@]}"; do
    require_file "${run_paths[$index]}"
    require_file "${se_negative[$index]}"
    require_file "${se_positive[$index]}"
done

verify_checkpoint inventory \
    "$work_root/input_manifest.tsv" "$work_root/software_versions.txt"
run_stage inventory record_inventory
verify_checkpoint prefreesurfer \
    "$session_root/T1w/T1w_acpc_dc_restore.nii.gz" \
    "$session_root/T1w/T2w_acpc_dc_restore.nii.gz"
run_stage prefreesurfer run_prefreesurfer
verify_checkpoint masked_atlas \
    "$session_root/MNINonLinear/registration_qc.json" \
    "$session_root/MNINonLinear/xfms/acpc_dc2standard.nii.gz"
run_stage masked_atlas repair_atlas_registration
verify_checkpoint freesurfer \
    "$session_root/T1w/$session/surf/lh.white" \
    "$session_root/T1w/$session/surf/rh.white"
run_stage freesurfer run_freesurfer
verify_checkpoint postfreesurfer \
    "$session_root/MNINonLinear/Native/$session.L.sphere.reg.native.surf.gii" \
    "$session_root/MNINonLinear/Native/$session.R.sphere.reg.native.surf.gii"
run_stage postfreesurfer run_postfreesurfer
for index in "${!run_names[@]}"; do
    verify_checkpoint "fmri_volume_${run_names[$index]}" \
        "$session_root/MNINonLinear/Results/${run_names[$index]}/${run_names[$index]}.nii.gz"
    run_stage "fmri_volume_${run_names[$index]}" run_fmri_volume "$index"
    verify_checkpoint "fmri_surface_${run_names[$index]}" \
        "$session_root/MNINonLinear/Results/${run_names[$index]}/${run_names[$index]}_Atlas.dtseries.nii"
    run_stage "fmri_surface_${run_names[$index]}" run_fmri_surface "$index"
done
verify_checkpoint multirun_fix \
    "$session_root/MNINonLinear/Results/rfMRI_REST_CONCAT/rfMRI_REST_CONCAT_Atlas_hp${high_pass_seconds}_clean.dtseries.nii" \
    "$session_root/MNINonLinear/Results/rfMRI_REST_CONCAT/rfMRI_REST_CONCAT_Atlas_hp${high_pass_seconds}_clean_vn.dscalar.nii"
run_stage multirun_fix run_fix
verify_checkpoint prepare_msmall \
    "$session_root/MNINonLinear/Results/rfMRI_REST_CONCAT/rfMRI_REST_CONCAT_Atlas_hp${high_pass_seconds}_clean_vn.dscalar.nii"
run_stage prepare_msmall prepare_msmall
verify_checkpoint msmall \
    "$session_root/MNINonLinear/Native/$session.L.sphere.${output_registration}_InitialReg_2_d${ica_dimension}_${method}.native.surf.gii" \
    "$session_root/MNINonLinear/Native/$session.R.sphere.${output_registration}_InitialReg_2_d${ica_dimension}_${method}.native.surf.gii"
run_stage msmall run_msmall
verify_checkpoint dedrift \
    "$session_root/MNINonLinear/Native/$session.L.sphere.${output_registration}.native.surf.gii" \
    "$session_root/MNINonLinear/Native/$session.R.sphere.${output_registration}.native.surf.gii"
run_stage dedrift run_dedrift
verify_checkpoint validate "$work_root/complete"
run_stage validate finalize
