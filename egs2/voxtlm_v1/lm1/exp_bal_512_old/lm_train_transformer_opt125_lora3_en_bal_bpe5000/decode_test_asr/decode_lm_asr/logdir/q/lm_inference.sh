#!/bin/bash
cd /projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.lm_inference --batch_size 1 --ngpu 1 --data_path_and_name_and_type dump/raw/test_512/text.asr,text,text --key_file exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/keys.${SLURM_ARRAY_TASK_ID}.scp --output_dir exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/output.${SLURM_ARRAY_TASK_ID} --token_type bpe --bpemodel data/en_bal_token_list/bpe_unigram5000/bpe.model --lm_train_config exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/config.yaml --lm_file exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/valid.acc.ave.pth --config conf/decode_lm_asr.yaml 
EOF
) >exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 -m espnet2.bin.lm_inference --batch_size 1 --ngpu 1 --data_path_and_name_and_type dump/raw/test_512/text.asr,text,text --key_file exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/keys.${SLURM_ARRAY_TASK_ID}.scp --output_dir exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/output.${SLURM_ARRAY_TASK_ID} --token_type bpe --bpemodel data/en_bal_token_list/bpe_unigram5000/bpe.model --lm_train_config exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/config.yaml --lm_file exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/valid.acc.ave.pth --config conf/decode_lm_asr.yaml  ) &>>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/lm_inference.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/q/done.168612.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -A p32222 -p gengpu --gres=gpu:a100:1 --constraint=sxm --mem 200G -c 16 --time 48:00:00 --error /home/ywn1043/yibo/err1.err  --output /home/ywn1043/yibo/out1.out  --open-mode=append -e exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/q/lm_inference.log -o exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/q/lm_inference.log --array 1-8 /projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1/exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/q/lm_inference.sh >>exp_bal_512//lm_train_transformer_opt125_lora3_en_bal_bpe5000/decode_test_asr/decode_lm_asr/logdir/q/lm_inference.log 2>&1
