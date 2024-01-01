#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
src_lang=es # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh
version=c # c or t (please refer to cvss paper for details)

 . utils/parse_options.sh || exit 1;

# base url for download commonvoice
# cv_data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/${src_lang}.tar.gz
# wget "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-15.0-2023-09-08/cv-corpus-15.0-2023-09-08-zh-CN.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20231113%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231113T200022Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=a22515d07d4e4b950b375ddd5660d9150f05ad91c30c3c1b7c76e258f0dc853e8cfd71c43b3a9953483d8988a74d984c36ff1dc55b53acb4cfde20791f96dba024e189be3dc0eb915e0cb84e67037c1ad26deae44a9ded6a8b1a7262153dd072b7698ffdc0ab51e69f13e0fe1569255e28d938cb418e82695090fc672115a793ff33e70c5bbb9a47f26fbafaf8adfb0b92d47c3a6023185ef6f09cc0070c435b455aa77f3a8094239d77069939a6da36cb817389e04d83eff660774939cb7108600585af4c61ea9ad26d1100fffc646c965d0cb074ecc371596b369dacf7ac821bc62ba0ceb12c537602b5924d49620286166cf30e7a84183fce630f36d82379"

cv_data_url="https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-4-2019-12-10/es.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20231127%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231127T054657Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=41460dfab93aa1f2ed3e4c7ed6ce4f983c84c84cca508d026b1a58ba6410279b73e2097e6f88310894cf3a25857d8e23f2a2b09e7e5c69c950bda3d29b400876f9d08cc3a93a5f1060e13e8086581bb7668a90c6d7ace6bdfb06924afbda30037ee27247d3a39f48cab756f30ef059b10e61a677fd387e8205a7c57a8f5c6dfde3d05bdeba09d30dc1cafe420eec7a3bcc7a6b3a1b3ef0ab0e2b4a159c2a4c7cbfdef947d8e17117bc1cf6750a745cea5cf2b3d8f814314613302cb614b1a3a7f2ba73f0ea257189909f4b7e10bee0701e0d2496eb1f3fd2fe4643e2f00bda56030b888790e86e6ae6a265f1770058dc1e604d11df9a6ce15f6f87412b9a61c8"
cvss_data_url=https://storage.googleapis.com/cvss/cvss_${version}_v1.0/cvss_${version}_${src_lang}_en_v1.0.tar.gz
# https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-4-2019-12-10/zh-CN.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20231113%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231113T195935Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=09bcc663bda918f2caccc32d40daf664ad58d15c0a40448dd4ca9117da2f0e65b0690914d98b92cdff09299569a1f3d3ae0f3ff0dabe9b6492ffd4a590f29d1197ce17fea712c548d96b15b46b20e26d04844fd6102e3a43df5d86b2a0e9c1ee9e622e8153c60eae9c8053aafb3956a2c9311186217b60ebbbc786b5f29e014ea725f67aea4a47596f2ee656afd1d45a6f5a3c7030663788433ed2bccc034a3ef47a24cca2ea8aec346daca7f3becbbdaa0d6f86f975879575599a3fbfcbe96f7ddb9e9f4781bc5673324bf3b51e05597e2055b9a35ef7002c0e4be3a26e4316fb77907f29ea89c5e2a852269bd89fd9e073f07d9f6b35cd582b844e44495df6
# https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-15.0-2023-09-08/cv-corpus-15.0-2023-09-08-zh-CN.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20231113%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231113T200022Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=a22515d07d4e4b950b375ddd5660d9150f05ad91c30c3c1b7c76e258f0dc853e8cfd71c43b3a9953483d8988a74d984c36ff1dc55b53acb4cfde20791f96dba024e189be3dc0eb915e0cb84e67037c1ad26deae44a9ded6a8b1a7262153dd072b7698ffdc0ab51e69f13e0fe1569255e28d938cb418e82695090fc672115a793ff33e70c5bbb9a47f26fbafaf8adfb0b92d47c3a6023185ef6f09cc0070c435b455aa77f3a8094239d77069939a6da36cb817389e04d83eff660774939cb7108600585af4c61ea9ad26d1100fffc646c965d0cb074ecc371596b369dacf7ac821bc62ba0ceb12c537602b5924d49620286166cf30e7a84183fce630f36d82379
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${CVSS}
if [ -z "${CVSS}" ]; then
    log "Fill the value of 'CVSS' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Download data to ${CVSS}"
    log "Prepare source data from commonvoice 4.0"
    # mkdir -p ${CVSS}/commonvoice4/${src_lang}
    # local/download_and_untar.sh ${CVSS}/commonvoice4/${src_lang} ${cv_data_url} ${src_lang}.tar.gz
    mkdir -p ${CVSS}/${src_lang}_en-${version}
    local/download_and_untar.sh ${CVSS}/${src_lang}_en-${version} ${cvss_data_url} cvss_${version}_${src_lang}_en_v1.0.tar.gz
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for commonvoice and cvss"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "train" "test" "dev"; do

        log "Prepare Commonvoice ${part}"
        if [ "${part}" = train ]; then
            local/cv_data_prep.pl \
                "${CVSS}/commonvoice4/${src_lang}" \
                validated data/"validated_${src_lang}" ${src_lang}
            mv data/"validated_${src_lang}" data/train_"${src_lang}"
        else
            local/cv_data_prep.pl \
                "${CVSS}/commonvoice4/${src_lang}" \
                ${part} data/"${part}_${src_lang}" ${src_lang}
        fi

        log "Prepare CVSS ${part}"
        python local/cvss_data_prep.py \
            --datadir "${CVSS}/${src_lang}_en-${version}" \
            --subset ${part} \
            --dest data/"${part}_${src_lang}" \
            --src_lang ${src_lang}

        ln -sf text.en data/"${part}_${src_lang}"/text
        ln -sf wav.scp.en data/"${part}_${src_lang}"/wav.scp

        utt_extra_files="wav.scp.${src_lang} wav.scp.en text.${src_lang} text.en"
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" data/${part}_${src_lang}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
