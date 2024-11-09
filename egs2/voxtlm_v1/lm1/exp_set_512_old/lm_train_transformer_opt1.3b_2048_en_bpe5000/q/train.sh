#!/bin/bash
cd /projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.lm_train --ngpu 4 --use_preprocessor true --bpemodel data/en_token_list/bpe_unigram5000/bpe.model --token_type bpe --token_list data/en_token_list/bpe_unigram5000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/dev/lm_text,text,text --valid_shape_file exp_set_512//lm_stats_en_bpe5000/valid/text_shape.bpe --fold_length 150 --resume true --output_dir exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000 --config conf/train_transformer_opt1.3b_2048.yaml --train_data_path_and_name_and_type exp_set_512//lm_stats_en_bpe5000/splits32/lm_train.txt,text,text --train_shape_file exp_set_512//lm_stats_en_bpe5000/splits32/text_shape.bpe --multiple_iterator true --ngpu 4 --multiprocessing_distributed True 
EOF
) >exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 -m espnet2.bin.lm_train --ngpu 4 --use_preprocessor true --bpemodel data/en_token_list/bpe_unigram5000/bpe.model --token_type bpe --token_list data/en_token_list/bpe_unigram5000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/dev/lm_text,text,text --valid_shape_file exp_set_512//lm_stats_en_bpe5000/valid/text_shape.bpe --fold_length 150 --resume true --output_dir exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000 --config conf/train_transformer_opt1.3b_2048.yaml --train_data_path_and_name_and_type exp_set_512//lm_stats_en_bpe5000/splits32/lm_train.txt,text,text --train_shape_file exp_set_512//lm_stats_en_bpe5000/splits32/text_shape.bpe --multiple_iterator true --ngpu 4 --multiprocessing_distributed True  ) &>>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
echo '#' Accounting: end_time=$time2 >>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
echo '#' Finished at `date` with status $ret >>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log
[ $ret -eq 137 ] && exit 100;
touch exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/q/done.269972
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -A p32222 -p gengpu --gres=gpu:a100:4 --constraint=sxm --mem 200G -c 16 --time 48:00:00 --error /home/ywn1043/yibo/err1.err  --job-name exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/train.log --output /home/ywn1043/yibo/out1.out  --open-mode=append -e exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/q/train.log -o exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/q/train.log  /projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1/exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/q/train.sh >>exp_set_512//lm_train_transformer_opt1.3b_2048_en_bpe5000/q/train.log 2>&1
