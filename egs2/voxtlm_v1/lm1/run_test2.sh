#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_bal"
valid_set="dev"
test_sets="test_0.01"

nbpe=5000
km_dir="" #Add pretrained km_directory path
lm_config=conf/train_transformer_opt350_qlora_8bit.yaml
lm_inference_asr_config=conf/decode_lm_asr.yaml
lm_inference_tts_config=conf/decode_lm_tts.yaml

./lm.sh \
    --stage 7 \
    --stop_stage 9 \
    --num_splits_lm 32 \
    --nj 32 \
    --ngpu 1 \
    --expdir exp_set/ \
    --gpu_inference true \
    --inference_nj 4 \
    --lang en \
    --token_type bpe \
    --nbpe "${nbpe}" \
    --bpe_nlsyms data/nlsyms.txt \
    --bpe_train_text "data/${train_set}/bpe_text" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_lm valid.acc.ave.pth \
    --km_dir "${km_dir}" \
    --lm_inference_asr_config "${lm_inference_asr_config}" \
    --lm_inference_tts_config "${lm_inference_tts_config}" \
    --lm_test_text_asr dump/raw/${test_sets}/text.asr \
    --lm_test_text_tts dump/raw/${test_sets}/text.tts \
    --lm_test_text_textlm dump/raw/${test_sets}/text.textlm \
    --lm_test_text_speechlm dump/raw/${test_sets}/text.speechlm "$@"
