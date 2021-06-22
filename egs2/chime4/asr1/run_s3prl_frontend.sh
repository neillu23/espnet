#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
valid_set=dt05_multi_isolated_1ch_track
test_sets="\
dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
"
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml


## APC
opts="--expdir exp_s3prl_apc"
asr_conf=train_asr_conformer_s3prl_frontend_apc_1_1; opts=${opts:-}" --feats_normalize null"

## CPC
# opts="--expdir exp_s3prl_cpc"
# asr_conf=train_asr_conformer_s3prl_frontend_cpc_1_1; opts=${opts:-}" --feats_normalize null"

## Wav2vec2
# opts="--expdir exp_s3prl_wav2vec2"
# asr_conf=train_asr_conformer_s3prl_frontend_wav2vec2_1_1; opts=${opts:-}" --feats_normalize null"

## Hubert
# opts="--expdir exp_s3prl_hubert"
# asr_conf=train_asr_conformer_s3prl_frontend_hubert_1_1; opts=${opts:-}" --feats_normalize null"

./asr.sh \
    --stage 3 --stop-stage 10 \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type char                      \
    --lang en \
    --ngpu 1 \
    --nbpe 500 \
    --max_wav_duration 30 \
    --asr_config "conf/tuning_s3prl/${asr_conf}.yaml" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --bpe_train_text "data/${train_set}/text" \
    ${opts} \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
