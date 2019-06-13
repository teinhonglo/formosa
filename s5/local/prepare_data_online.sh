#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

set -euo pipefail
data_root=data/online
dataset=eval0
use_text=false

echo "$0 $@"

. ./cmd.sh
. ./utils/parse_options.sh

corpus=$1

if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the LibriSpeech corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing online evaluation data"

# have to remvoe previous files to avoid filtering speakers according to cmvn.scp and feats.scp
rm -rf   $data_root/$dataset
mkdir -p $data_root/$dataset

#
# make utt2spk, wav.scp and text
#

cur_data_root=$data_root/$dataset
touch $cur_data_root/utt2spk
touch $cur_data_root/wav.scp
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $y' \; | dos2unix > $cur_data_root/utt2spk
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $x' \; | dos2unix > $cur_data_root/wav.scp

cat $cur_data_root/utt2spk > $cur_data_root/spk2utt

if $use_text; then
  find $corpus -name "*.txt" -exec sh -c 'x={}; y=${x%.txt}; printf "%s " $y; cat $x'    \; | dos2unix | sed 's/\/Text\//\/Wav\//' > $cur_data_root/text
fi

utils/fix_data_dir.sh $cur_data_root


echo "Data preparation completed."

