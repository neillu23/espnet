./lm.sh --stage 7 --stop_stage 9 --num_splits_lm 32 --nj 32 --ngpu 2 --expdir exp_set/ --gpu_inference true --inference_nj 32 --lang en --token_type bpe --nbpe 5000 --bpe_nlsyms data/nlsyms.txt --bpe_train_text data/train_bal/bpe_text --lm_config conf/train_transformer_opt350_2.yaml --train_set train_bal --valid_set dev --test_sets test --inference_lm valid.acc.best.pth --km_dir  --lm_inference_asr_config conf/decode_lm_asr.yaml --lm_inference_tts_config conf/decode_lm_tts.yaml --lm_test_text_asr dump/raw/test/text.asr --lm_test_text_tts dump/raw/test/text.tts --lm_test_text_textlm dump/raw/test/text.textlm --lm_test_text_speechlm dump/raw/test/text.speechlm --stage 7 --stage 7 "$@"; exit $?
