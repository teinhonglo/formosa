#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

set -euo pipefail

skip_lm=false

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

corpus=$1
dataset=$2

if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the LibriSpeech corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing augumentation train set"

# have to remvoe previous files to avoid filtering speakers according to cmvn.scp and feats.scp
rm -rf    data/$dataset
mkdir -p  data/$dataset

#
# make utt2spk, wav.scp and text
#
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $y' \; | dos2unix > data/$dataset/utt2spk
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $x' \; | dos2unix > data/$dataset/wav.scp
find $corpus -name *.txt -exec sh -c 'x={}; y=${x%.txt}; printf "%s " $y; cat $x'    \; | dos2unix | sed 's/\/Text\//\/Wav\//' > data/$dataset/text
sed -i 's/　//g' data/$dataset/text

#
# fix data format
#
for x in $dataset; do
    # fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
    # duplicate entries and so on). Also, it regenerates the spk2utt from
    # utt2sp
    utils/fix_data_dir.sh data/$x
done

if ! $skip_lm ; then
  echo "cp data/train/text data/local/train/text for language model training"
  cat data/$dataset/text | awk '{$1=""}1;' | awk '{$1=$1}1;' > data/local/train/text_vol2
  cat data/local/train/text data/local/train/text_vol2 > data/local/train/text_vol1_2
fi

echo "Data $dataset incorporiate completed."

