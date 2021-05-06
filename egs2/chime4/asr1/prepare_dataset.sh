#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

cond=simu    # simu, real

enh="_2021_beam_tasnet_70ep"
enh="_mvdr"

datasets="dt05_${cond}${enh} et05_${cond}${enh}"
datasets="dt05_${cond}${enh}"
datasets="dt05 et05"
#datasets="dt05"

#for x in dt05_${cond} et05_${cond}; do
#    mkdir -p data/${x}${enh}
#    cp ../../../egs2/chime4/enh1/data/${x}_isolated_6ch_track/{spk2utt,text,utt2spk} data/${x}${enh}/
#    cp ../../../egs2/chime4/enh1/exp/enh_train_enh_beamformer_no_wpe_raw/enhanced_${x}_isolated_6ch_track_multich${enh}/spk1.scp data/${x}${enh}/wav.scp
#    # cp ../../../egs2/chime4/enh1/exp/enh_train_enh_beamformer_no_wpe_raw/enhanced_${x}${enh}/spk1.scp data/${x}${enh}/wav.scp
#    utils/validate_data_dir.sh --no-feats data/${x}${enh}
#done
for x in ${datasets}; do
    xx="${x}_${cond}${enh}"
    mkdir -p dump/raw/${xx}
    cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/dump/raw/${x}_${cond}_isolated_6ch_track/{spk2utt,text,utt2spk} dump/raw/${xx}/
    #cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/exp/enh_train_enh_mc_conv_tasnet_snr_raw/enhanced_${x}_${cond}_isolated_6ch_track/spk1.scp data/${xx}/wav.scp
    #cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/exp_6ch_rotated/enh_train_enh_miso_conv_tasnet_snr_raw/enhanced_${x}_${cond}_isolated_6ch_track/spk1.scp data/${xx}/wav.scp
    if [ "$cond" = "simu" ]; then
        cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_enh/egs2/chime4/enh1/exp/enh_train_enh_beamformer_no_wpe_raw/enhanced_${x}_${cond}_mvdr/spk1.scp dump/raw/${xx}/wav.scp
    elif [ "$cond" = "real" ]; then
        cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_enh/egs2/chime4/enh1/exp/enh_train_enh_beamformer_no_wpe_raw/enhanced_${x}_${cond}_isolated_6ch_track_multich/spk1.scp dump/raw/${xx}/wav.scp
    fi
    echo "raw" > dump/raw/${xx}/feats_type
    # cp ../../../egs2/chime4/enh1/exp/enh_train_enh_beamformer_no_wpe_raw/enhanced_${x}/spk1.scp data/${xx}/wav.scp
    utils/validate_data_dir.sh --no-feats dump/raw/${xx}
done
