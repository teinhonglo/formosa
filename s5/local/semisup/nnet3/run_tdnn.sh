#!/bin/bash

#    This is the standard "tdnn" system, built in nnet3.

set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
min_seg_len=1.55
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
ihm_gmm=tri3      # Only relevant if $use_ihm_ali is true, the name of the gmm-dir in
                  # the ihm directory that is to be used for getting alignments.
num_threads_ubm=8
nnet3_affix=_cleaned  # cleanup affix for exp dirs, e.g. _cleaned
tdnn_affix=  #affix for TDNN directory e.g. "a" or "b", in case we change the configuration.

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
remove_egs=true
relu_dim=850
num_epochs=3

# set common_egs_dir to use previously dumped egs.
common_egs_dir=
srand=0
exp_root=
data_root=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
<<WORD
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"

# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --min-seg-len $min_seg_len \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set
WORD
if $use_ihm_ali; then
  gmm_dir=$exp_root/${ihm_gmm}
  ali_dir=$exp_root/${ihm_gmm}_ali_${train_set}_sp_comb_ihmdata
  lores_train_data_dir=$data_root/${train_set}_ihmdata_sp_comb
  maybe_ihm="IHM "
  dir=$exp_root/nnet3${nnet3_affix}/tdnn${tdnn_affix}_sp_ihmali
else
  gmm_dir=$exp_root/${gmm}
  ali_dir=$exp_root/${gmm}_ali_${train_set}_sp_comb
  lores_train_data_dir=$data_root/${train_set}_sp_comb
  maybe_ihm=
  dir=$exp_root/nnet3${nnet3_affix}/tdnn${tdnn_affix}_sp
fi


train_data_dir=$data_root/${train_set}_sp_hires_comb
train_ivector_dir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 11 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning perturbed, short-segment-combined ${maybe_ihm}data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd --num_threads 5" \
         ${lores_train_data_dir} data/lang $gmm_dir $ali_dir
fi

[ ! -f $ali_dir/ali.1.gz ] && echo  "$0: expected $ali_dir/ali.1.gz to exist" && exit 1

if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$relu_dim
  relu-batchnorm-layer name=tdnn2 dim=$relu_dim input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 dim=$relu_dim input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn4 dim=$relu_dim input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn5 dim=$relu_dim input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
	echo "split"
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi
	
	steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$cuda_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=400000 \
	--trainer.optimization.minibatch-size 512,128 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=6 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --dir=$dir  || exit 1;
fi
graph_dir=$dir/graph_poco
if [ $stage -le 14 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  # diff.
  # utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco_unk $dir $graph_dir
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco $dir $graph_dir
fi

if [ $stage -le 15 ]; then
  rm $dir/.error || true 2>/dev/null
  for decode_set in dev eval; do
      (
      decode_dir=${dir}/decode_${decode_set}
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir $exp_root/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
         $graph_dir $data_root/${decode_set}_hires $decode_dir
      ) &
  done
  wait;
  if [ -f $dir/.error ]; then
    echo "$0: error detected during decoding"
    exit 1
  fi
fi


exit 0;
