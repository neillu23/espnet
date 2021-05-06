#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

cond=real    # simu, real
#enh="_mvdr"
#enh="_tasnet"
#enh="_psm"
#enh="_1ch"  # original isolated 1ch data
#enh="_beamformit_5mics"

#enh="_mvdr_new_5mic"
#enh="_2021_mc_beam_tasnet"
#enh="_2021_mctasnet"
#enh="_2021_miso_tasnet"
#enh="_2021_beam_tasnet_masking"
#enh="_2021_beam_tasnet_70ep_vad_masking"
#enh="_2021_beam_tasnet_70ep_psm_masking"
#enh="_2021_beam_tasnet_70ep"
#enh="_2021_miso_tasnet_tfloss"
#enh="_2021_miso_tasnet_tfloss_fintuned"
#enh="_2021_miso_tasnet_tfloss_scratch"
#enh="_2021_miso_tasnet_tfloss_scratch_beam_tasnet_psm_masking"
#enh="_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking"
#enh="_2021_fasnet_snr_4ms"
#enh="_2021_fasnet_snr"
#enh="_2021_fasnet_tac_snr_4ms"
#enh="_2021_fasnet_tac_snr"
#enh="_2021_beam_tasnet_5ep_siso_tasnet_vad_masking"
#enh="_2021_beam_tasnet_5ep_siso_tasnet_psm_masking"
#enh="_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad2"
#enh="_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad3"
#enh="_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad1"
#enh="_2021_2ep_siso_tasnet_newdata"
#enh="_2021_mctasnet_2.0tfloss"
enh="_2021_beam_tasnet_2ep_siso_tasnet_newdata_vad_masking"

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
    mkdir -p data/${xx}
    #cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/dump/raw/${x}_${cond}_isolated_1ch_track/{spk2utt,text,utt2spk} data/${xx}/
    #cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/exp_newdata/enh_train_enh_conv_tasnet_denoising_raw/enhanced_${x}_${cond}_isolated_1ch_track/spk1.scp data/${xx}/wav.scp
    cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/dump/raw/${x}_${cond}_isolated_6ch_track/{spk2utt,text,utt2spk} data/${xx}/
    #cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/exp_6ch_rotated/enh_train_enh_miso_conv_tasnet_snr_2.0tfloss_256win_stable_raw/enhanced_${x}_${cond}_isolated_6ch_track/spk1.scp data/${xx}/wav.scp
    #cp /mnt/lustre/sjtu/home/cdl54/workspace/asr/develop/espnet_waspaa/egs2/chime4/enh1/exp/enh_train_enh_miso_fasnet_snr_raw/enhanced_${x}_${cond}_isolated_6ch_track/spk1.scp data/${xx}/wav.scp
    cp /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/dump/raw/${x}_${cond}_isolated_6ch_track/beam_tasnet_New2ep.pre_vad_masking.mvdr.nowpe.normalize/spk1.scp data/${xx}/wav.scp
    # cp ../../../egs2/chime4/enh1/exp/enh_train_enh_beamformer_no_wpe_raw/enhanced_${x}/spk1.scp data/${xx}/wav.scp
    utils/validate_data_dir.sh --no-feats data/${xx}
done

for x in ${datasets}; do
    xx="${x}_${cond}${enh}"
    utils/copy_data_dir.sh data/${xx} data-fbank/${xx}
    utils/fix_data_dir.sh data-fbank/${xx}
    #utils/copy_data_dir.sh data/${xx} data-stft/${xx}
    steps/make_fbank_pitch.sh --nj 8 --cmd "run.pl" --write_utt2num_frames true \
        data-fbank/${xx} exp/make_fbank/${xx} fbank
done

for x in ${datasets}; do
    xx="${x}_${cond}${enh}"
    feat_recog_dir=dump/${xx}/deltafalse; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "run.pl" --nj 4 --do_delta false \
        data-fbank/${xx}/feats.scp data-fbank/tr05_multi_noisy_si284_sp/cmvn.ark exp/dump_feats/recog/${xx} \
        ${feat_recog_dir}
done

for x in ${datasets}; do
    xx="${x}_${cond}${enh}"
    feat_recog_dir=dump/${xx}/deltafalse; mkdir -p ${feat_recog_dir}
    data2json.sh --feat ${feat_recog_dir}/feats.scp \
        --nlsyms data/lang_1char/non_lang_syms.txt \
        data/${xx} data/lang_1char/tr05_multi_noisy_si284_sp_units.txt \
        > ${feat_recog_dir}/data.json
done
