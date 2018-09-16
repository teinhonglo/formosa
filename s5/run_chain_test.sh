#!/bin/bash

set -euo pipefail

# This script is modified based on mini_librispeech/s5/local/nnet3/run_ivector_common.sh

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
test_set=test

nj=20

nnet3_affix=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

test_set=$1
dir=$2

 
if [ $stage -le 1 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  utils/data/copy_data_dir.sh data/${test_set} data/${test_set}_hires
  steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${test_set}_hires || exit 1;
  steps/compute_cmvn_stats.sh data/${test_set}_hires || exit 1;
  utils/fix_data_dir.sh data/${test_set}_hires || exit 1;
  # create MFCC data dir without pitch to extract iVector
  utils/data/limit_feature_dim.sh 0:39 data/${test_set}_hires data/${test_set}_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/${test_set}_hires_nopitch || exit 1;
fi

if [ $stage -le 2 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
    data/${test_set}_hires_nopitch exp/nnet3/extractor \
    exp/nnet3/ivectors_${test_set}
fi


if [ $stage -le 3 ]; then
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 10 --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_$test_set \
      $graph_dir data/${test_set}_hires $dir/decode_${test_set} || exit 1;
fi
