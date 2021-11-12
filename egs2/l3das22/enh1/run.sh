#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

train_set=train_multich   # for training MC-TasNet / MISO-TasNet
valid_set=dev_multich
test_sets=dev_multich 


./enh.sh --audio_format wav \
    --stage 1 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --spk_num 1 \
    --enh_config conf/tuning/train_enh_mc_conv_tasnet_post_beamformer_ci_sdr.yaml \
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
