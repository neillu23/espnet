#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test_qut test"


sample_rate=16000

./enh.sh \
    --stage 1 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 3 \
    --spk_num 1 \
    --ref_channel 0 \
    --audio_format wav \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
