#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/tuning/train_ecapa_Vox12_xlsr.yaml
# spk_config=conf/train_RawNet3.yaml

train_set="voxceleb2cat_train_proc_audio_no_sil_train"
valid_set="voxceleb2cat_train_proc_audio_no_sil_val"
test_sets="voxceleb1_test"

./spk.sh \
<<<<<<< HEAD
    --inference_model valid.eer.ave.pth \
    --feats_type ${feats_type} \
=======
    --ngpu 2 \
    --feats_type raw \
<<<<<<< HEAD
    --inference_model valid.eer.ave.pth \
=======
>>>>>>> 0762440c1fc048ade7296629197edec5722cdccb
>>>>>>> d7b00f390a87a91757889fa0a09e57f138791d32
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    "$@"
