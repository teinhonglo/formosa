#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

set -euo pipefail

corpus=$1

. ./cmd.sh
. ./utils/parse_options.sh

if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the LibriSpeech corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing train and test data"

# have to remvoe previous files to avoid filtering speakers according to cmvn.scp and feats.scp
rm -rf   data/online/eval0
mkdir -p data/online/eval0

#
# make utt2spk, wav.scp and text
#

cur_data_root=data/online/eval0
touch $cur_data_root/utt2spk
touch $cur_data_root/wav.scp
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $y' \; | dos2unix > $cur_data_root/utt2spk
find $corpus -name *.wav -exec sh -c 'x={}; y=${x%.wav}; printf "%s %s\n"     $y $x' \; | dos2unix > $cur_data_root/wav.scp
utils/fix_data_dir.sh $cur_data_root

echo "Data preparation completed."

