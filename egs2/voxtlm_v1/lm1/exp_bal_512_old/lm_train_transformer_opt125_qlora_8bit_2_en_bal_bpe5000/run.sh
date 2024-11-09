./lm.sh --stage 1 --stop_stage 9 --num_splits_lm 1 --nj 16 --ngpu 1 --expdir exp_bal_512/ --gpu_inference true --inference_nj 8 --lang en_bal --token_type bpe --nbpe 5000 --bpe_nlsyms data/nlsyms.txt --bpe_train_text data/train_bal/bpe_text --lm_config conf/train_transformer_opt125_qlora_8bit_2.yaml --train_set train_bal --valid_set dev --test_sets test_512 --inference_lm valid.acc.ave.pth --km_dir  --lm_inference_asr_config conf/decode_lm_asr.yaml --lm_inference_tts_config conf/decode_lm_tts.yaml --lm_test_text_asr dump/raw/test_512/text.asr --lm_test_text_tts dump/raw/test_512/text.tts --lm_test_text_textlm dump/raw/test_512/text.textlm --lm_test_text_speechlm dump/raw/test_512/text.speechlm --stage 7 --stage 7 "$@"; exit $?
