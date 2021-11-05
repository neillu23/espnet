#!/bin/bash

. ./path.sh

list_set="tr05_simu_isolated_6ch_track"

for x in $list_set; do
    mkdir -p data/${x}_rotated/.tmp

    mix-mono-wav-scp.py data/local/data/${x}_wav.CH{1,3,4,5,6}.scp > data/${x}_rotated/.tmp/wav.CH1.scp
    mix-mono-wav-scp.py data/local/data/${x}_spk1_wav.CH{1,3,4,5,6}.scp > data/${x}_rotated/.tmp/spk1.CH1.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" data/${x}_rotated/.tmp/spk1.CH1.scp > data/${x}_rotated/.tmp/noise1.CH1.scp

    mix-mono-wav-scp.py data/local/data/${x}_wav.CH{3,4,5,6,1}.scp > data/${x}_rotated/.tmp/wav.CH3.scp
    mix-mono-wav-scp.py data/local/data/${x}_spk1_wav.CH{3,4,5,6,1}.scp > data/${x}_rotated/.tmp/spk1.CH3.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" data/${x}_rotated/.tmp/spk1.CH3.scp > data/${x}_rotated/.tmp/noise1.CH3.scp

    mix-mono-wav-scp.py data/local/data/${x}_wav.CH{4,5,6,1,3}.scp > data/${x}_rotated/.tmp/wav.CH4.scp
    mix-mono-wav-scp.py data/local/data/${x}_spk1_wav.CH{4,5,6,1,3}.scp > data/${x}_rotated/.tmp/spk1.CH4.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" data/${x}_rotated/.tmp/spk1.CH4.scp > data/${x}_rotated/.tmp/noise1.CH4.scp

    mix-mono-wav-scp.py data/local/data/${x}_wav.CH{5,6,1,3,4}.scp > data/${x}_rotated/.tmp/wav.CH5.scp
    mix-mono-wav-scp.py data/local/data/${x}_spk1_wav.CH{5,6,1,3,4}.scp > data/${x}_rotated/.tmp/spk1.CH5.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" data/${x}_rotated/.tmp/spk1.CH5.scp > data/${x}_rotated/.tmp/noise1.CH5.scp

    mix-mono-wav-scp.py data/local/data/${x}_wav.CH{6,1,3,4,5}.scp > data/${x}_rotated/.tmp/wav.CH6.scp
    mix-mono-wav-scp.py data/local/data/${x}_spk1_wav.CH{6,1,3,4,5}.scp > data/${x}_rotated/.tmp/spk1.CH6.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" data/${x}_rotated/.tmp/spk1.CH6.scp > data/${x}_rotated/.tmp/noise1.CH6.scp

    for i in 1 3 4 5 6; do
        sed -i -e "s#^\(\w\+\) #\1_CH${i}first #g" data/${x}_rotated/.tmp/wav.CH${i}.scp
        sed -i -e "s#^\(\w\+\) #\1_CH${i}first #g" data/${x}_rotated/.tmp/spk1.CH${i}.scp
        sed -i -e "s#^\(\w\+\) #\1_CH${i}first #g" data/${x}_rotated/.tmp/noise1.CH${i}.scp
    done

    cat data/${x}_rotated/.tmp/wav.CH{1,3,4,5,6}.scp | sort > data/${x}_rotated/wav.scp
    cat data/${x}_rotated/.tmp/spk1.CH{1,3,4,5,6}.scp | sort > data/${x}_rotated/spk1.scp
    cat data/${x}_rotated/.tmp/noise1.CH{1,3,4,5,6}.scp | sort > data/${x}_rotated/noise1.scp
done
