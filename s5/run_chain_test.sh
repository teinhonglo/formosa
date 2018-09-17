#!/bin/bash

set -euo pipefail

# This script is modified based on mini_librispeech/s5/local/nnet3/run_ivector_common.sh

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=-1
test_set=test

nj=20

nnet3_affix=
data_dir=/share/corpus/MATBN_GrandChallenge/NER-Trs-Vol1-Eval
data_root=data/online

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

test_set=$1
dir=$2

rm local/score.sh
ln -s score_online.sh local/score.sh
 
if [ $stage -le -1 ]; then
  # Data Preparation
  echo "$0: Data Preparation"
  local/prepare_data_online.sh $data_dir --data-root $data_root --dataset $test_set || exit 1;
fi

if [ $stage -le 1 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  utils/data/copy_data_dir.sh $data_root/${test_set} $data_root/${test_set}_hires
  steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
      --cmd "$train_cmd" $data_root/${test_set}_hires || exit 1;
  steps/compute_cmvn_stats.sh $data_root/${test_set}_hires || exit 1;
  utils/fix_data_dir.sh $data_root/${test_set}_hires || exit 1;
  # create MFCC data dir without pitch to extract iVector
  utils/data/limit_feature_dim.sh 0:39 $data_root/${test_set}_hires $data_root/${test_set}_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh $data_root/${test_set}_hires_nopitch || exit 1;
fi

if [ $stage -le 2 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    $data_root/${test_set}_hires_nopitch exp/nnet3/extractor \
    exp/nnet3/ivectors_${test_set}
fi

if [ $stage -le 3 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

if [ $stage -le 4 ]; then
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_$test_set \
      $dir/graph $data_root/${test_set}_hires $dir/decode_${test_set} || exit 1;
fi


rm local/score.sh
ln -s score_real.sh local/score.sh

echo "$0 Done."
exit 0;
