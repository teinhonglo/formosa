#!/bin/bash

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on aishell2 to rm.
#
# Model preparation: The last layer (prefinal and output layer) from
# already-trained aishell2 model is removed and 3 randomly initialized layer
# (new tdnn layer, prefinal, and output) are added to the model.
#
# Training: The transferred layers are retrained with smaller learning-rate,
# while new added layers are trained with larger learning rate using rm $data_root.
# The chain config is as in run_tdnn_5n.sh and the result is:
#System tdnn_5n tdnn_aishell2_ami_1a
#WER      2.71     1.68
set -e

data_root=data/transfer
exp_root=exp/transfer
# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
xent_regularize=0.1

# configs for transfer learning
src_mdl=../../aishell2/s5/exp/chain/tdnn_1d_all_sp/final.mdl # Input chain model
                                                   # trained on source $data_rootset (aishell2).
                                                   # This model is transfered to the target domain.

src_mfcc_config=../../aishell2/s5/conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features for ivector and DNN training
                                                  # in the source domain.
src_ivec_extractor_dir=../../aishell2/s5/exp/chain/extractor  # Source ivector extractor dir used to extract ivector for
                         # source $data_root. The ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in the source model training.

common_egs_dir=
primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.

nnet3_affix=_online_aishell2
chain_affix=_online_aishell2
train_set=train_vol1_2_3b_cleaned

nj=20
gmm=tri7b_cleaned  # Expect GMM model in $exp/$gmm for alignment
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

required_files="$src_mfcc_config $src_mdl"
use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is specified "
    echo "but ivector is not used in training the source model '$src_mdl'."
  else
    required_files="$required_files $src_ivec_extractor_dir/final.dubm $src_ivec_extractor_dir/final.mat $src_ivec_extractor_dir/final.ie"
    use_ivector=true
  fi
else
  if [ $ivector_dim -gt 0 ]; then
    echo "$0: ivector is used in training the source model '$src_mdl' but no "
    echo " --src-ivec-extractor-dir option as ivector dir for source model is specified." && exit 1;
  fi
fi

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f." && exit 1;
  fi
done

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 4" if you have already
# run those things.

gmm_dir=$exp_root/../${gmm}
ali_dir=$exp_root/../${gmm}_sp_ali
lat_dir=$exp_root/chain${chain_affix}/${gmm}_sp_lats
treedir=$exp_root/chain${chain_affix}/tree_bi${tree_affix}
dir=$exp_root/chain${chain_affix}/tdnn_aishell2_ami_1a
lang=data/lang_chain_ais2fms

train_data_dir=$data_root/${train_set}_sp_hires
lores_train_data_dir=$data_root/${train_set}_sp

local/online/run_extractor.sh  --stage $stage --nj $nj \
                               --mfcc-config $src_mfcc_config \
                               --ivector-dim $ivector_dim \
                               --nnet3-affix "$nnet3_affix" \
			       --data-root $data_root \
			       --train-set $train_set \
			       --exp-root $exp_root \
                               --extractor $src_ivec_extractor_dir || exit 1;

train_ivector_dir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
								  
if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd --num-threads 3" $lores_train_data_dir \
    data/lang $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz 2>/dev/null || true # save space
fi

if [ $stage -le 5 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -r $lang 2>/dev/null || true
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 6 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --leftmost-questions-truncate -1 \
    --cmd "$train_cmd" 5000 $lores_train_data_dir $lang $ali_dir $treedir || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "$0: Create neural net configs using the xconfig parser for";
  echo " generating new layers, that are specific to rm. These layers ";
  echo " are added to the transferred part of the aishell2 network.";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  tdnnf-layer name=tdnnf18-target $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3 input=tdnnf17.noop
  linear-component name=prefinal-l dim=256 $linear_opts
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_mdl - \| \
    nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: generate egs for chain to train new model on ami $data_root set."
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$cuda_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$train_ivector_dir" \
    --chain.xent-regularize $xent_regularize \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width 150,110,90 \
    --trainer.num-chunk-per-minibatch=128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.005 \
    --trainer.optimization.final-effective-lrate=0.0005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1;
fi

graph_affix=_13
graph_dir=$dir/graph${graph_affix}
if [ $stage -le 9 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang${graph_affix}_test $dir $graph_dir
fi

if [ $stage -le 10 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir $exp_root/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir $data_root/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
