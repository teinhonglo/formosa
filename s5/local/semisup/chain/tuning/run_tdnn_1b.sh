#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

set -e
set -o pipefail

# This is fisher chain recipe for training a model on a subset of around
# 100-300 hours of supervised data.
# local/semisup/run_50k.sh and local/semisup/run_100k.sh show how to call this.

# configs for 'chain'
mic=ihm
use_ihm_ali=false
stage=0
train_stage=-10
get_egs_stage=-10
min_seg_len=1.55
exp_root=exp/$mic/semisup_20k
data_root=data/$mic/semisup

nj=30
tdnn_affix=_1b
train_set=train_sup
ivector_train_set=   # dataset for training i-vector extractor
ivector_transform_type=pca
num_threads_ubm=8

nnet3_affix=  # affix for nnet3 dir -- relates to i-vector used
chain_affix=_1b  # affix for chain dir
tree_affix=_a
cleaned_affix=
gmm=tri3  # Expect GMM model in $exp/$gmm for alignment

# Neural network opts
xent_regularize=0.1


# training options
num_epochs=4
remove_egs=false
common_egs_dir=   # if provided, will skip egs generation
common_treedir=   # if provided, will skip the tree building stage

decode_iter=
cleand_affix=
comb_affix=_hires_comb
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

lang=data/lang_chain

#if [ ! -z $ivector_train_set ]; then
#   ivector_train_set=${ivector_train_set}${cleand_affix}
#fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
local/nnet3/run_ivector_common_semi.sh  --stage $stage \
                                           --mic $mic \
                                           --nj $nj \
                                           --min-seg-len $min_seg_len \
                                           --train_set $train_set \
					                       --ivector-train-set "$ivector_train_set" \
                                           --gmm $gmm \
                                           --num-threads-ubm $num_threads_ubm \
                                           --nnet3-affix "$nnet3_affix" \
                                           --ivector-transform-type "$ivector_transform_type" \
                                           --data_root $data_root \
                                           --exp_root $exp_root

# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats_semi.sh --stage $stage \
                                          --mic $mic \
                                          --nj $nj \
                                          --data_root $data_root \
                                          --min-seg-len $min_seg_len \
                                          --use-ihm-ali $use_ihm_ali \
                                          --train_set $train_set										  
										  
if $use_ihm_ali; then
  gmm_dir=$exp_root/${ihm_gmm}
  ali_dir=$exp_root/${ihm_gmm}_ali_${train_set}_sp_comb_ihmdata
  lores_train_data_dir=$data_root/${train_set}_ihmdata_sp_comb
  tree_dir=$exp_root/chain${chain_affix}/tree_${tree_affix}_ihmdata
  lat_dir=$exp_root/chain${chain_affix}/${gmm}_${train_set}_sp_comb_lats_ihmdata
  dir=$exp_root/chain${chain_affix}/tdnn${tdnn_affix}_sp_ihmali
  # note: the distinction between when we use the 'ihmdata' suffix versus
  # 'ihmali' is pretty arbitrary.
else
  # gmm_dir=$exp_root/$gmm   # used to get training lattices (for chain supervision)
  gmm_dir=$exp_root/$gmm
  ali_dir=$exp_root/${gmm}_ali_${train_set}_sp_comb
  lores_train_data_dir=$data_root/${train_set}_sp_comb
  # tree_dir=$exp_root/chain${chain_affix}/tree_${tree_affix}
  tree_dir=$exp_root/chain${chain_affix}/tree_bi${tree_affix}
  # lat_dir=$exp_root/chain${chain_affix}/${gmm}_${train_set}_sp_unk_lats  # training lattices directory
  lat_dir=$exp_root/chain${chain_affix}/${gmm}_${train_set}_sp_comb_lats
  # dir=$exp_root/chain${chain_affix}/tdnn${tdnn_affix}_sp
  dir=$exp_root/chain${chain_affix}/tdnn${tdnn_affix}_sp_bi
fi

train_data_dir=$data_root/${train_set}_sp${comb_affix}
train_ivector_dir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp${comb_affix}
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

for f in $gmm_dir/final.mdl $lores_train_data_dir/feats.scp \
   $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 12 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning perturbed, short-segment-combined ${maybe_ihm}data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
     ${lores_train_data_dir} data/lang $gmm_dir $ali_dir
fi

[ ! -f $ali_dir/ali.1.gz ] && echo  "$0: expected $ali_dir/ali.1.gz to exist" && exit 1

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
							--generate-ali-from-lats true \
							${lores_train_data_dir} data/lang $gmm_dir $lat_dir
  # diff.
  # --generate-ali-from-lats true
  # data/lang_unk							
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  #if [ $lang/L.fst -nt data/lang/L.fst ]; then
  #  echo "$0: $lang already exists, not overwriting it; continuing"
  #else
  #  echo "$0: $lang already exists and seems to be older than data/lang..."
  #  echo " ... not sure what to do.  Exiting."
  #  exit 1;
  #fi
  
  rm -rf $lang
  cp -r data/lang $lang	
  #cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ -z "$common_treedir" ]; then
    if [ $stage -le 15 ]; then
    # Build a tree using our new topology.  We know we have alignments for the
    # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
    # those.
   
	if [ -f $tree_dir/final.mdl ]; then
	  echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
	  exit 1;
	fi
  
    steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4200 ${lores_train_data_dir} $lang $ali_dir $tree_dir
	# diff. fisher $lat_dir, ami $ali_dir  
  
	fi
else
    tree_dir=$common_treedir
fi	  

xent_regularize=0.1

if [ $stage -le 16 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=450
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=450
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=450
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=450

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn7 dim=450 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn7 dim=450 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 17 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  mkdir -p $dir/egs
  touch $dir/egs/.nodelete # keep egs around when that run dies.
	
   steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$cuda_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;	
fi

#graph_dir=$dir/graph_${LM}
graph_dir=$dir/graph_poco
if [ $stage -le 18 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  # diff.
  # utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco_unk $dir $graph_dir
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco $dir $graph_dir
fi

if [ $stage -le 19 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd --num-threads 2" --num-threads 4 \
          --online-ivector-dir $exp_root/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir $data_root/${decode_set}_hires $dir/decode_${decode_set} || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1;
  fi
fi

exit 0;
