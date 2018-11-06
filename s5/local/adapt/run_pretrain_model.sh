#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 50 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we assume there is data/train/text, from which
# we will exclude the utterances contained in the unsupervised set.
# We use all 300 hours of semi-supervised data for i-vector extractor training.

# This differs from run_100k.sh, which uses only 100 hours supervised data for 
# both i-vector extractor training and LM training.

. ./cmd.sh
. ./path.sh 

set -euo pipefail


stage=2
nj=30 # number of parallel jobs,
mic=ihm
corpus=aspire
pretrain_model_dir=../../pretrain_model/$corpus/exp
train_sup_dir=train_sup
train_unsup_dir=train_unsup80k
ada_dir=adapt
exp_root=exp/$mic/adapt
data_root=data/$mic/$ada_dir
dir=$exp_root/$corpus/chain/tdnn_7b

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

. utils/parse_options.sh

if [ ! -d $pretrain_model_dir ]; then
  echo "$0, Expected $pretrain_model_dir directory exist."
  exit 0
fi

if [ $stage -le 1 ]; then
  mkdir -p $exp_root
  mkdir -p $exp_root/$corpus
  cp -r $pretrain_model_dir/chain $exp_root/$corpus
  cp -r $pretrain_model_dir/nnet3 $exp_root/$corpus
fi

exp_root=$exp_root/$corpus

if [ $stage -le 2 ]; then
  for datadir in dev eval ; do
    utils/copy_data_dir.sh $data_root/../$datadir $data_root/${datadir}_hires_${corpus}
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_${corpus}.conf \
                                  --cmd "$train_cmd" $data_root/${datadir}_hires_${corpus}
    steps/compute_cmvn_stats.sh $data_root/${datadir}_hires_${corpus}
    utils/fix_data_dir.sh $data_root/${datadir}_hires_${corpus}
  done

  for datadir in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
        $data_root/${datadir}_hires_${corpus} $exp_root/nnet3/extractor \
        $exp_root/nnet3/ivectors_${datadir}_hires  
  done
fi

graph_dir=$dir/graph
if [ $stage -le 3 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 4 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd --num_threads 4" \
          --online-ivector-dir $exp_root/nnet3/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " --stage 2 \
         $graph_dir $data_root/${decode_set}_hires_${corpus} $dir/decode_${decode_set}  || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0

