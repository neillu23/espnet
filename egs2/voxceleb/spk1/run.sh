#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/tuning/train_ecapa_Vox12_xlsr.yaml

train_set="voxceleb2cat_train_proc_audio_no_sil_train"
valid_set="voxceleb2cat_train_proc_audio_no_sil_val"
test_sets="voxceleb1_test"
feats_type="raw"

./spk.sh \
    --inference_model valid.eer.ave.pth \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    "$@"
