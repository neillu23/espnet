#!/bin/bash


. ./path.sh
. ./cmd.sh

echo "${*}"

nj=1
datadir=dump/raw/dt05_simu_isolated_6ch_track
outdir=$PWD
test_only=False

#outdir=dump/raw/dev_noreverb/array_selection.no_snr_selection.wpe #.tasnet_post_proc
#outdir=dump/raw/dev_noreverb/array_selection.no_snr_selection.mpdr.wpe.tasnet_post_proc
#outdir=dump/raw/test/array_selection.no_snr_selection.wpe.tasnet_post_proc

########################
# Parse some arguments #
########################
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --datadir)
        datadir="$2"
        shift # past argument
        shift # past value
        ;;
        --outdir)
        outdir="$2"
        shift # past argument
        shift # past value
        ;;
        --test_only)
        test_only="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

mkdir -p "${outdir}"

if [ $nj -gt 1 ]; then
    for j in $(seq ${nj}); do
        mkdir -p ${datadir}/split${nj}/output.${j}
    done
    utils/split_scp.pl "${datadir}/wav.scp" "$(for j in $(seq ${nj}); do echo ${datadir}/split${nj}/output.${j}/wav.${j}.scp; done)"
    utils/split_scp.pl "${datadir}/spk1.scp" "$(for j in $(seq ${nj}); do echo ${datadir}/split${nj}/output.${j}/spk1.${j}.scp; done)"
    ${decode_cmd} --gpu 1 JOB=1:${nj} ${datadir}/split${nj}/eval_beamformer.JOB.log \
        python ./eval_beamformer.py \
            --wavscp "$datadir"/split${nj}/output.JOB/wav.JOB.scp \
            --spkscp "$datadir"/split${nj}/output.JOB/spk1.JOB.scp \
            --outdir "$outdir" \
            --test_only "$test_only" \
            "$@"

    if [ "${test_only,,}" = "false" ]; then
        for mode in input enhanced; do
            for metric in ESTOI PESQ SAR SDR SIR SI_SNR STOI; do
                cat "$(for j in $(seq ${nj}); do echo ${datadir}/split${nj}/output.${j}/${mode}_${metric}_spk1; done)" > ${datadir}/split${nj}/${mode}_${metric}_spk1

                <${datadir}/split${nj}/${mode}_${metric}_spk1 \
                    awk 'BEGIN{sum=0}
                    {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                    END{print sum/NR}' \
                > "${datadir}/split${nj}/result_${mode}_${metric,,}.txt"
            done
        done
    fi

else

    python ./eval_beamformer.py \
        --wavscp "$datadir"/wav.scp \
        --spkscp "$datadir"/spk1.scp \
        --outdir "$outdir" \
        --test_only "$test_only" \
        "$@"

    if [ "${test_only,,}" = "false" ]; then
        for mode in input enhanced; do
            for metric in ESTOI PESQ SAR SDR SIR SI_SNR STOI; do
                <"$outdir"/${mode}_${metric}_spk1 \
                    awk 'BEGIN{sum=0}
                    {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                    END{print sum/NR}' \
                > "$outdir"/result_${mode}_${metric,,}.txt
            done
        done
    fi
fi
