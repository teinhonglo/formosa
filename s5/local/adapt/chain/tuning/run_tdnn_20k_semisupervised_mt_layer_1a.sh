#!/bin/bash
# Apache 2.0
set -u -e -o pipefail

stage=0   # Start from -1 for supervised seed system training
train_stage=-100
nj=80
test_nj=50
min_seg_len=1.55

# The following 3 options decide the output directory for semi-supervised 
# chain system
# dir=${exp_root}/chain${chain_affix}/tdnn${tdnn_affix}
mic=ihm
exp_root=exp/$mic/semisup_20k
data_root=data/$mic/semisup
chain_affix=_semi20k_20k_80k    # affix for chain dir
tdnn_affix=_semisup_ts_mt_1a

# Datasets -- Expects data/$supervised_set and data/$unsupervised_set to be
# present
supervised_set=train_sup50k
unsupervised_set=train_unsup100k_250k

# Input seed system
sup_chain_dirs=($exp_root/chain_semi20k_80k/tdnn_1b_sp_bi \
                $exp_root/chain_semi20k_80k/tdnn_1a_sp_bi \
				$exp_root/chain_semi20k_80k/tdnn_1c_sp_bi \
				$exp_root/chain_semi20k_80k/tdnn_1d_sp_bi)

num_sys=${#sup_chain_dirs[@]}
sup_chain_dir=sup_chain_dir[0]
sup_lat_dir=$exp_root/chain_semi50k_100k_250k/tri4a_train_sup50k_unk_lats  # lattices for supervised set
sup_tree_dir=$exp_root/chain_semi50k_100k_250k/tree_bi_a  # tree directory for supervised chain system
ivector_root_dir=$exp_root/nnet3_semi50k_100k_250k  # i-vector extractor root directory

# Semi-supervised options
supervision_weights=1.5,1,1,1,1     # Weights for supervised, unsupervised data egs.
                                    # Can be used to scale down the effect of unsupervised data
                                    # by using a smaller scale for it e.g. 1.0,0.3
lm_weights=3,2,3,2,3,2,3,2  # Weights on phone counts from supervised, unsupervised data for denominator FST creation

sup_egs_dir=   # Supply this to skip supervised egs creation
unsup_egs_dir=  # Supply this to skip unsupervised egs creation
unsup_egs_opts=  # Extra options to pass to unsupervised egs creation

# Neural network opts
xent_regularize=0.1

decode_iter=  # Iteration to decode with
phn_affix=

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

sup_chain_dir_base=`basename $sup_chain_dir`
tdnn_semi=${sup_chain_dir_base}${tdnn_affix}${phn_affix}


# The following can be replaced with the versions that do not model
# UNK using phone LM. $sup_lat_dir should also ideally be changed.
unsup_decode_lang=data/lang_test_poco_ex86k_unk
unsup_decode_graph_affix=_poco_ex86k_unk
test_lang=data/lang_test_poco
test_graph_affix=_poco

unsup_rescore_lang=${unsup_decode_lang}_big

dir=$exp_root/chain${chain_affix}/${tdnn_semi}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

supervised_set_perturbed=${supervised_set}_sp
unsupervised_set_perturbed=${unsupervised_set}_sp

sup_ivector_dir=$ivector_root_dir/ivectors_${supervised_set_perturbed}_hires_comb

graphdir=$sup_chain_dir/graph${unsup_decode_graph_affix}

for f in $data_root/${supervised_set_perturbed}/feats.scp \
  $data_root/${supervised_set_perturbed}_hires_comb/feats.scp \
  $ivector_root_dir/extractor/final.ie $sup_ivector_dir/ivector_online.scp \
  $sup_lat_dir/lat.1.gz $sup_tree_dir/ali.1.gz \
  $unsup_decode_lang/G.fst; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done


frame_subsampling_factor=3
#if [ -f $sup_chain_dir/frame_subsampling_factor ]; then
#  frame_subsampling_factor=$(cat $sup_chain_dir/frame_subsampling_factor)
#fi
cmvn_opts=$(cat $sup_chain_dir/cmvn_opts) || exit 1

diff $sup_tree_dir/tree $sup_chain_dir/tree || { echo "$0: $sup_tree_dir/tree and $sup_chain_dir/tree differ"; exit 1; }
# Uncomment the following lines if you need to build new tree using both
# supervised and unsupervised data. This may help if amount of
# supervised data used to train the seed system tree is very small.
# unsupervised data

# tree_affix=bi_semisup_a
# treedir=$exp_root/chain${chain_affix}/tree_${tree_affix}
# if [ -f $treedir/final.mdl ]; then
#   echo "$0: $treedir/final.mdl exists. Remove it and run again."
#   exit 1
# fi
#
# if [ $stage -le 9 ]; then
#   # This is usually 3 for chain systems.
#   echo $frame_subsampling_factor > \
#     $sup_chain_dir/best_path_${unsupervised_set_perturbed}_big/frame_subsampling_factor
#
#   # This should be 1 if using a different source for supervised data alignments.
#   # However alignments in seed tree directory have already been sub-sampled.
#   echo $frame_subsampling_factor > \
#     $sup_tree_dir/frame_subsampling_factor
#
#   # Build a new tree using stats from both supervised and unsupervised data
#   steps/nnet3/chain/build_tree_multiple_sources.sh \
#     --use-fmllr false --context-opts "--context-width=2 --central-position=1" \
#     --frame-subsampling-factor $frame_subsampling_factor \
#     7000 $lang \
#     data/${supervised_set_perturbed}_hires_comb \
#     ${sup_tree_dir} \
#     data/${unsupervised_set_perturbed}_hires_comb \
#     $chaindir/best_path_${unsupervised_set_perturbed} \
#     $treedir || exit 1
# fi
#
# sup_tree_dir=$treedir   # Use the new tree dir for further steps

# Train denominator FST using phone alignments from
# supervised and unsupervised data
if [ $stage -le 10 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${sup_tree_dir} ${sup_chain_dir}/best_path_${unsupervised_set_perturbed}_big \
	${sup_tree_dir} ${sup_chain_dirs[1]}/best_path_${unsupervised_set_perturbed}_big \
	${sup_tree_dir} ${sup_chain_dirs[2]}/best_path_${unsupervised_set_perturbed}_big \
	${sup_tree_dir} ${sup_chain_dirs[3]}/best_path_${unsupervised_set_perturbed}_big \
    $dir
fi

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $sup_tree_dir/tree |grep num-pdfs|awk '{print $2}')
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
  output-layer name=output input=prefinal-chain include-log-softmax=false dim=$num_targets max-change=1.5
  
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
  
  # We use separate outputs for supervised and unsupervised data
  # so we can properly track the train and valid objectives.
  output name=output-0 input=output.affine
  output name=output-1 input=output.affine
  output name=output-2 input=output.affine
  output name=output-3 input=output.affine
  output name=output-4 input=output.affine
  output name=output-0-xent input=output-xent.log-softmax
  output name=output-1-xent input=output-xent.log-softmax
  output name=output-2-xent input=output-xent.log-softmax
  output name=output-3-xent input=output-xent.log-softmax
  output name=output-4-xent input=output-xent.log-softmax
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

# Get values for $model_left_context, $model_right_context
. $dir/configs/vars

left_context=$model_left_context
right_context=$model_right_context

egs_left_context=$(perl -e "print int($left_context + $frame_subsampling_factor / 2)")
egs_right_context=$(perl -e "print int($right_context + $frame_subsampling_factor / 2)")

if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set_perturbed}
  frames_per_eg=$(cat $sup_chain_dir/egs/info/frames_per_eg)

  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $egs_left_context --right-context $egs_right_context \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor $frame_subsampling_factor \
               --frames-per-eg $frames_per_eg \
               --frames-per-iter 1500000 \
               --cmvn-opts "$cmvn_opts" \
               --online-ivector-dir $sup_ivector_dir \
               --generate-egs-scp true \
               $data_root/${supervised_set_perturbed}_hires_comb $dir \
               $sup_lat_dir $sup_egs_dir
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsup_frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when
                      # creating numerator supervision
lattice_prune_beam=4  # beam for pruning the lattices prior to getting egs
                        # for unsupervised data
tolerance=1   # frame-tolerance for chain training

for i in `seq 0 $[num_sys-1]`; do
  echo "$0 ${sup_chain_dirs[$i]}"
  sup_chain_dir=${sup_chain_dirs[$i]};
  unsup_lat_dir=${sup_chain_dir}/decode_${unsupervised_set_perturbed}_big
  unsup_affix=`basename $sup_chain_dir | cut -d "_" -f2`
  unsup_egs_dirs[$i]=$dir/egs_${unsupervised_set_perturbed}${phn_affix}_${unsup_affix}
  unsup_egs_dir=${unsup_egs_dirs[$i]}
  if [ $stage -le 13 ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
        utils/create_split_dir.pl \
         /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$unsup_egs_dir/storage $unsup_egs_dir/storage
      fi
      mkdir -p $unsup_egs_dir
      touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

      echo "$0: generating egs from the unsupervised data"
      steps/nnet3/chain/get_egs.sh \
        --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
        --left-tolerance $tolerance --right-tolerance $tolerance \
        --left-context $egs_left_context --right-context $egs_right_context \
        --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
        --frame-subsampling-factor $frame_subsampling_factor \
        --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
        --lattice-prune-beam "$lattice_prune_beam" \
        --deriv-weights-scp $sup_chain_dir/best_path_${unsupervised_set_perturbed}_big/weights.scp \
        --online-ivector-dir $ivector_root_dir/ivectors_${unsupervised_set_perturbed}_hires_comb \
        --generate-egs-scp true $unsup_egs_opts \
        $data_root/${unsupervised_set_perturbed}_hires_comb $dir \
        $unsup_lat_dir $unsup_egs_dir
  fi
done

comb_egs_dir=$dir/comb_egs${phn_affix}
if [ $stage -le 14 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 128 \
    --lang2weight $supervision_weights 5 \
    $sup_egs_dir ${unsup_egs_dirs[@]} $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  # This is to skip stages of den-fst creation, which was already done.
  train_stage=-4
fi

if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir "$comb_egs_dir" \
    --cmd "$cuda_cmd" \
    --feat.online-ivector-dir $sup_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir $data_root/${supervised_set_perturbed}_hires_comb \
    --tree-dir $sup_tree_dir \
    --lat-dir $sup_lat_dir \
    --dir $dir || exit 1;
fi
test_graph_affix=${test_graph_affix}${phn_affix}
test_graph_dir=$dir/graph${test_graph_affix}
if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${test_lang} $dir $test_graph_dir
fi

if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
    for decode_set in dev eval; do
      (
	  #num_jobs=`cat $data_root/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd --num-threads 2" --num-threads 5 \
          --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $test_graph_dir $data_root/${decode_set}_hires $dir/decode${test_graph_affix}_${decode_set} || exit 1;
      ) || touch $dir/.error &
    done
  wait;
  if [ -f $dir/.error ]; then
    echo "$0: Decoding failed. See $dir/decode${test_graph_affix}_*/log/*"
    exit 1
  fi
fi

exit 0;
