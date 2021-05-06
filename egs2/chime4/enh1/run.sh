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

#train_set=tr05_simu_isolated_6ch_track
train_set=tr05_simu_isolated_6ch_track_rotated     # for training MC-TasNet / MISO-TasNet
valid_set=dt05_simu_isolated_6ch_track
#test_sets="et05_simu_isolated_6ch_track"
test_sets="et05_simu_isolated_6ch_track dt05_real_isolated_6ch_track et05_real_isolated_6ch_track"
#test_sets="dt05_real_isolated_6ch_track et05_real_isolated_6ch_track"
#test_sets="et05_simu_isolated_6ch_track"

#valid_set=dt05_real_isolated_6ch_track
#test_sets="et05_real_isolated_6ch_track"

#train_set=tr05_simu_isolated_1ch_track
#train_set=tr05_simu_isolated_1ch_track_plus_simu_35k
#valid_set=dt05_simu_isolated_1ch_track
#test_sets="et05_simu_isolated_1ch_track"
#test_sets="dt05_real_isolated_1ch_track et05_real_isolated_1ch_track"

./enh.sh --audio_format wav \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ref_channel 4 \
    --rotate_channel 4 \
    --fs ${sample_rate} \
    --ngpu 4 \
    --spk_num 1 \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 2 --stop-stage 2" \
    --enh_config conf/tuning/train_enh_miso_conv_tasnet_snr_2.0tfloss_256win_stable.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.snr.best.pth" \
    "$@"

#    --enh_config conf/tuning/train_enh_beamformer_mvdr_snr.yaml \
#    --enh_config conf/tuning/train_enh_miso_conv_tasnet_snr.yaml \
#    --enh_config conf/tuning/train_enh_miso_conv_tasnet_snr_tfloss.yaml \
#    --enh_config conf/tuning/train_enh_miso_conv_tasnet_snr_tfloss_256win.yaml \
#    --enh_config conf/tuning/train_enh_miso_conv_tasnet_snr_2.0tfloss_256win.yaml \
#    --enh_config conf/tuning/train_enh_miso_conv_tasnet_snr_2.0tfloss_256win_stable.yaml \
#    --enh_config conf/tuning/train_enh_mc_conv_tasnet_snr.yaml \
#    --enh_config conf/tuning/train_enh_conv_tasnet_denoising.yaml \
#    --enh_config conf/tuning/train_enh_conv_tasnet_componentloss.yaml \
#    --enh_config conf/tuning/train_enh_dprnn_tasnet.yaml \
#    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
#    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
#    --enh_config conf/tuning/train_enh_blstm_tf.yaml \
