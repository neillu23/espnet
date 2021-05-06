#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

. ./path.sh
. ./cmd.sh

nj=20
num_spk=1
fs=16000
ref_channel=4

ref_scp=
inf_scp=
out=result_pesq.txt

. utils/parse_options.sh


tmpdir=$(mktemp -d pesq_score-XXXX)
chmod 755 "$tmpdir"
echo "Creating temporary directory: $tmpdir"
logdir="$PWD"/log
mkdir -p "$logdir"

_nj=$(min "${nj}" "$(<${ref_scp} wc -l)")
split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${tmpdir}/ref.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${ref_scp}" ${split_scps}
split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${tmpdir}/inf.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${inf_scp}" ${split_scps}


out="$(realpath $out)"
for spk in $(seq $num_spk); do
    ${decode_cmd} JOB=1:"${_nj}" "${logdir}/compute_$(basename $tmpdir)".JOB.log \
        compute-pesq-score.sh \
            --ref_channel ${ref_channel} \
            --nostrict True \
            --fs ${fs} \
            --out "$tmpdir"/PESQ.JOB.spk${spk} \
            "$tmpdir"/ref.JOB.scp \
            "$tmpdir"/inf.JOB.scp \

    echo -n > "$tmpdir"/PESQ.spk${spk}.tmp
    for j in $(seq ${_nj}); do
        cat "$tmpdir"/PESQ.${j}.spk${spk} >> "$tmpdir"/PESQ.spk${spk}.tmp
    done
    sort "$tmpdir"/PESQ.spk${spk}.tmp > "$tmpdir"/PESQ.spk${spk}
done

paste "$(for spk in $(seq ${num_spk}); do echo "${tmpdir}/PESQ.spk${spk}" ; done)" > "${out}"

awk 'BEGIN{sum=0}
    {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
    END{print sum/NR}' "${out}"

rm -r $tmpdir
