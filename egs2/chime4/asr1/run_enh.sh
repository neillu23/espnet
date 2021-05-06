#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
valid_set=dt05_multi_isolated_1ch_track
test_sets="dt05_simu_mvdr et05_simu_mvdr dt05_real_mvdr et05_real_mvdr"


use_word_lm=false
word_vocab_size=65000

./asr.sh --audio_format wav                \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type char                      \
    --feats_type raw                       \
    --speed_perturb_factors '0.9 1.0 1.1'  \
    --asr_config conf/train_asr_transformer_specaug.yaml  \
    --inference_config conf/decode_asr_transformer.yaml   \
    --lm_config conf/tuning/train_lm.yaml  \
    --inference_lm valid.loss.best.pth     \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
