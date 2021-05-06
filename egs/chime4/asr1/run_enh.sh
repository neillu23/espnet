#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# data
chime4_data=/export/corpora4/CHiME4/CHiME3 # JHU setup
wsj0=/export/corpora5/LDC/LDC93S6B            # JHU setup
wsj1=/export/corpora5/LDC/LDC94S13B           # JHU setup

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr05_simu_mvdr # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
train_dev=dt05_real_psm
recog_set="${train_dev} et05_real_psm"

train_dev=dt05_simu_mvdr_new_5mic
recog_set="${train_dev} dt05_simu_mvdr_new_5mic_tasnet_postproc"

train_dev=dt05_simu_2021_mc_beam_tasnet
recog_set="${train_dev} et05_simu_2021_mc_beam_tasnet"

train_dev=dt05_real_2021_mctasnet
recog_set="${train_dev} et05_real_2021_mctasnet"

train_dev=dt05_simu_2021_beam_tasnet_masking
recog_set="${train_dev} et05_simu_2021_beam_tasnet_masking dt05_real_2021_beam_tasnet_masking et05_real_2021_beam_tasnet_masking"

train_dev=dt05_simu_2021_beam_tasnet_70ep_psm_masking
recog_set="${train_dev} et05_simu_2021_beam_tasnet_70ep_psm_masking dt05_real_2021_beam_tasnet_70ep_psm_masking et05_real_2021_beam_tasnet_70ep_psm_masking"

train_dev=dt05_simu_2021_beam_tasnet_70ep
recog_set="${train_dev} et05_simu_2021_beam_tasnet_70ep dt05_real_2021_beam_tasnet_70ep et05_real_2021_beam_tasnet_70ep"

train_dev=dt05_simu_2021_miso_tasnet_tfloss
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss dt05_real_2021_miso_tasnet_tfloss et05_real_2021_miso_tasnet_tfloss"

train_dev=dt05_simu_2021_miso_tasnet_tfloss_fintuned
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_fintuned dt05_real_2021_miso_tasnet_tfloss_fintuned et05_real_2021_miso_tasnet_tfloss_fintuned "

train_dev=dt05_simu_2021_miso_tasnet_tfloss_scratch
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_scratch dt05_real_2021_miso_tasnet_tfloss_scratch et05_real_2021_miso_tasnet_tfloss_scratch"

train_dev=dt05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_psm_masking
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_psm_masking dt05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_psm_masking et05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_psm_masking"

train_dev=dt05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking dt05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking et05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking"

train_dev=dt05_simu_2021_fasnet_snr_4ms
recog_set="${train_dev} et05_simu_2021_fasnet_snr_4ms dt05_real_2021_fasnet_snr_4ms et05_real_2021_fasnet_snr_4ms"

train_dev=dt05_simu_2021_fasnet_snr
recog_set="${train_dev} et05_simu_2021_fasnet_snr dt05_real_2021_fasnet_snr et05_real_2021_fasnet_snr"

train_dev=dt05_simu_2021_fasnet_tac_snr_4ms
recog_set="${train_dev} et05_simu_2021_fasnet_tac_snr_4ms dt05_real_2021_fasnet_tac_snr_4ms et05_real_2021_fasnet_tac_snr_4ms"

train_dev=dt05_simu_2021_fasnet_tac_snr
recog_set="${train_dev} et05_simu_2021_fasnet_tac_snr dt05_real_2021_fasnet_tac_snr et05_real_2021_fasnet_tac_snr"

train_dev=dt05_simu_2021_beam_tasnet_5ep_siso_tasnet_vad_masking
recog_set="${train_dev} et05_simu_2021_beam_tasnet_5ep_siso_tasnet_vad_masking dt05_real_2021_beam_tasnet_5ep_siso_tasnet_vad_masking et05_real_2021_beam_tasnet_5ep_siso_tasnet_vad_masking"

train_dev=dt05_simu_2021_beam_tasnet_5ep_siso_tasnet_psm_masking
recog_set="${train_dev} et05_simu_2021_beam_tasnet_5ep_siso_tasnet_psm_masking dt05_real_2021_beam_tasnet_5ep_siso_tasnet_psm_masking et05_real_2021_beam_tasnet_5ep_siso_tasnet_psm_masking"

train_dev=dt05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad2
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad2 dt05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad2 et05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad2"

train_dev=dt05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad1
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad1 dt05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad1 et05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad1"

train_dev=dt05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad3
recog_set="${train_dev} et05_simu_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad3 dt05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad3 et05_real_2021_miso_tasnet_tfloss_scratch_beam_tasnet_vad_masking_post_vad3"

train_dev=dt05_simu_2021_2ep_siso_tasnet_newdata
#recog_set="${train_dev} et05_simu_2021_2ep_siso_tasnet_newdata dt05_real_2021_2ep_siso_tasnet_newdata et05_real_2021_2ep_siso_tasnet_newdata"
recog_set="dt05_real_2021_2ep_siso_tasnet_newdata et05_real_2021_2ep_siso_tasnet_newdata"

train_dev=dt05_simu_2021_mctasnet_2.0tfloss
recog_set="${train_dev} et05_simu_2021_mctasnet_2.0tfloss dt05_real_2021_mctasnet_2.0tfloss et05_real_2021_mctasnet_2.0tfloss"

#recog_set="dt05_simu_2021_fasnet_snr_4ms dt05_simu_2021_fasnet_snr dt05_simu_2021_fasnet_tac_snr_4ms dt05_simu_2021_fasnet_tac_snr"

train_dev=dt05_simu_2021_beam_tasnet_2ep_siso_tasnet_newdata_vad_masking
recog_set="${train_dev} et05_simu_2021_beam_tasnet_2ep_siso_tasnet_newdata_vad_masking dt05_real_2021_beam_tasnet_2ep_siso_tasnet_newdata_vad_masking et05_real_2021_beam_tasnet_2ep_siso_tasnet_newdata_vad_masking"



#feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_1char/tr05_multi_noisy_si284_sp_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_pytorch_lm_word65000
expdir=exp/tr05_multi_noisy_si284_sp_pytorch_train_pytorch_transformer.sp
mkdir -p ${expdir}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=20

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0
        recog_model=model.last10.avg.best

        ${decode_cmd} JOB=1:${nj} --mem 10G ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
