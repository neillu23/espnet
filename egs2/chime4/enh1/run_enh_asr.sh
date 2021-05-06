#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.
extra_annotations=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/egs/chime4/asr1_multich/annotations

enh_model_path=pretrained_enh.pth
asr_model_path=pretrained_asr.pth

train_set=tr05_simu_isolated_6ch_track
#train_set=tr05_simu_isolated_6ch_track_rotated     # for training MC-TasNet / MISO-TasNet
#train_set=tr05_simu_isolated_6ch_track_tr05_real_isolated_6ch_track
train_set=tr05_simu_isolated_6ch_track_tr05_real_isolated_6ch_track_train_si284
#valid_set=dt05_simu_isolated_6ch_track
valid_set=dt05_multi_isolated_6ch_track
test_sets="et05_simu_isolated_6ch_track dt05_simu_isolated_6ch_track dt05_real_isolated_6ch_track et05_real_isolated_6ch_track"
#test_sets="dt05_real_isolated_6ch_track et05_real_isolated_6ch_track"

#train_set=tr05_simu_isolated_1ch_track
#train_set=tr05_simu_isolated_1ch_track_plus_simu_35k
#valid_set=dt05_simu_isolated_1ch_track
#test_sets="et05_simu_isolated_1ch_track"


./enh_asr.sh --audio_format wav \
    --lang "en" \
    --nbpe 5000 \
    --audio_format wav \
    --max_wav_duration 15 \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/tuning/train_lm.yaml \
    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup1e-4_specaug_tfloss.accum_grad.enh_real_prob0.7.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ref_channel 4 \
    --fs "${sample_rate}" \
    --ngpu 4 \
    --num_nodes 1 \
    --spk_num 1 \
    --use_signal_ref true \
    --use_signal_dereverb_ref false \
    --use_signal_noise_ref true \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 2 --stop-stage 2" \
    --srctexts "data/train_si284/text data/local/other_text/text" \
    --init_param "${enh_model_path}:encoder:enh_subclass.encoder ${enh_model_path}:separator:enh_subclass.separator ${enh_model_path}:decoder:enh_subclass.decoder ${asr_model_path}:encoder:asr_subclass.encoder ${asr_model_path}:decoder:asr_subclass.decoder ${asr_model_path}:ctc:asr_subclass.ctc" \
    --decode_joint_model valid.acc.ave_10best.pth \
    "$@"

#    --train_aux_set tr05_real_isolated_6ch_track \
#    --train_aux_set train_si284 \

#    --init_param "${enh_model_path}:encoder:enh_subclass.encoder ${enh_model_path}:separator:enh_subclass.separator ${enh_model_path}:decoder:enh_subclass.decoder ${asr_model_path}:encoder:asr_subclass.encoder ${asr_model_path}:decoder:asr_subclass.decoder ${asr_model_path}:ctc:asr_subclass.ctc" \

#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_1.0enh.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup1e-4_specaug_tfloss.accum_grad.enh_real_prob0.7.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup1e-4_specaug_tfloss.accum_grad.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup1e-4_specaug_tfloss.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup5e-5_specaug.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup1e-4_specaug.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup1e-3.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real_warmup.yaml \
#    --joint_config conf/tuning/train_enh_mc_conv_tasnet_asr_init_real.yaml \
