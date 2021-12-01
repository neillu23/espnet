#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_multich   # for training MC-TasNet / MISO-TasNet
valid_set=dev_multich
test_sets=dev_multich 


./asr_hfwav2vec2.sh \
    --lang en \
    --token_type char \
    --feats_normalize null \
    --lm_config conf/train_lm.yaml \
    --asr_config conf/train_asr_hfwav2vec2.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/train_nodev/text" "$@"
