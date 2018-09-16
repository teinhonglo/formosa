#!/bin/bash
#
# Copyright 2018, Yuan-Fu Liao, National Taipei University of Technology, yfliao@mail.ntut.edu.tw
#
# Before you run this recips, please apply, download and put or make a link of the corpus under this folder (folder name: "NER-Trs-Vol1").
# For more detail, please check:
# 1. Formosa Speech in the Wild (FSW) project (https://sites.google.com/speech.ntut.edu.tw/fsw/home/corpus)
# 2. Formosa Speech Recognition Challenge (FSW) 2018 (https://sites.google.com/speech.ntut.edu.tw/fsw/home/challenge)
. ./cmd.sh
. ./path.sh

stage=-2
train_stage=-10
num_jobs=20
data_dir=/share/corpus/MATBN_GrandChallenge/NER-Trs-Vol1-Eval
data_root=data/online
exp_root=exp
dir=
graph_dir=
# shell options
set -euo pipefail

. ./utils/parse_options.sh

# configure number of jobs running in parallel, you should adjust these numbers according to your machines
# data preparation
if [ $stage -le -2 ]; then
  # Data Preparation
  echo "$0: Data Preparation"
  local/prepare_data_online.sh $data_dir || exit 1;
fi

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

data_dir=eval0

# mfcc
if [ $stage -le -1 ]; then
  echo "$0: Making mfccs"
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $num_jobs $data_root/$data_dir || exit 1;
  steps/compute_cmvn_stats.sh $data_root/$data_dir || exit 1;
  utils/fix_data_dir.sh $data_root/$data_dir || exit 1;
fi

# mono
if [ $stage -le 0 ]; then

  # Monophone decoding
  (
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs --stage 3 \
    exp/mono/graph $data_root/$data_dir $exp_root/mono/decode_$data_dir
  )

fi

# tri1
if [ $stage -le 1 ]; then

  # decode tri1
  (
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/tri1/graph $data_root/$data_dir $exp_root/tri1/decode_$data_dir
  )

fi

# tri2
if [ $stage -le 2 ]; then

  # decode tri2
  (
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $num_jobs \
    exp/tri2/graph $data_root/$data_dir $exp_root/tri2/decode_$data_dir
  )

fi

# tri3a
if [ $stage -le 3 ]; then

  # decode tri3a
  (
  steps/decode.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
    exp/tri3a/graph $data_root/$data_dir $exp_root/tri3a/decode_$data_dir
  )

fi

# tri4
if [ $stage -le 4 ]; then

  # decode tri4a
  (
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
    exp/tri4a/graph $data_root/$data_dir $exp_root/tri4a/decode_$data_dir
  )

fi

# tri5
if [ $stage -le 5 ]; then

  # decode tri5
  (
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $num_jobs --config conf/decode.config \
     exp/tri5a/graph $data_root/$data_dir $exp_root/tri5a/decode_$data_dir || exit 1;
  )

fi

echo "$0: all done"

exit 0;
