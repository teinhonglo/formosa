#!/bin/bash
# _1b is as _1a, but different as follows
# 1) It uses librispeech phone set phones.txt and new lexicon generated using word pronunciation
#    in swj lexincon.txt. rm words, that are not presented in librispeech, are added as oov
#    in new lexicon.txt.
# 2) It uses librispeech tree-dir and generates new alignments and lattices for rm using
#    librispeech gmm model.
# 3) It also trains phone LM using weighted combination of alignemts from librispeech
#    and rm, which is used in chain denominator graph.
#    Since we use phone.txt from source dataset, this can be helpful in cases
#    where there is few training data in the target domain and some 4-gram phone
#    sequences have no count in the target domain.
# 4) It uses whole already-trained model and  does not replace the output layer
#    from already-trained model with new randomely initialized output layer and
#    re-train it using target dataset.


# This script uses weight transfer as a transfer learning method
# and use already trained model on librispeech and fine-tune the whole network using rm data
# while training the last layer (output layer) with higher learning-rate.
# The chain config is as run_tdnn_5n.sh and the result is:
# System tdnn_5n tdnn_librispeech_rm_1a tdnn_librispeech_rm_1b tdnn_librispeech_rm_1c
# WER      2.71     1.68            3.56          3.54
set -e

data_root=data/ihm/semisup
exp_root=exp/ihm/semisup_20k
# configs for 'chain'
stage=0
train_stage=-4
get_egs_stage=-10
nj=80
train_set=train_sup
gmm=
tdnn_affix=_1b
nnet3_affix=_online_librispeech
chain_affix=_online_librispeech

# configs for transfer learning
common_egs_dir=
primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, it fixed the paramters transferred from source.
                       # The learning-rate factor for new added layers is 1.0.
phone_lm_scales="1,10" # comma-separated list of positive integer multiplicities
                       # to apply to the different source data directories (used
                       # to give the RM data a higher weight).

# model and dirs for source model used for transfer learning
src_mdl=../../Librispeech/s5/exp/chain_cleaned/tdnn_1b_sp/final.mdl # Input chain model
                                                    # trained on source dataset (librispeech).
                                                    # This model is transfered to the target domain.

src_mfcc_config=../../Librispeech/s5/conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features for ivector and DNN training
                                                  # in the source domain.
src_ivec_extractor_dir=../../Librispeech/s5/exp/nnet3_cleaned/extractor  # Source ivector extractor dir used to extract ivector for
                         # source data and the ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in source model training.

src_lang=../../Librispeech/s5/data/lang # Source lang directory used to train source model.
                                # new lang dir for transfer learning experiment is prepared
                                # using source phone set phones.txt and lexicon.txt
                                # in src lang and dict dirs and words.txt in target lang dir.

src_dict=../../Librispeech/s5/data/local/dict_nosp  # dictionary for source dataset containing lexicon.txt,
                                            # nonsilence_phones.txt,...
                                            # lexicon.txt used to generate lexicon.txt for
                                            # src-to-tgt transfer.

src_gmm_dir=../../Librispeech/s5/exp/tri6b_cleaned # source gmm dir used to generate alignments
                                   # for target data.

src_tree_dir=../../Librispeech/s5/exp/chain_cleaned/tree_sp # chain tree-dir for src data;
                                         # the alignment in target domain is
                                         # converted using src-tree

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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

# dirs for src-to-tgt transfer learning experiment
lang_src_tgt=data/lang_librispeech_ami # This dir is prepared using phones.txt and lexicon from
                          # Librispeech and wordlist and G.fst from AMI.
lat_dir=$exp_root/chain${chain_affix}/${gmm}_${train_set}_sp_comb_lats_librispeech
dir=$exp_root/chain${chain_affix}/tdnn_librispeech_ami${tdnn_affix}

required_files="$src_mfcc_config $src_mdl $src_lang/phones.txt $src_dict/lexicon.txt $src_gmm_dir/final.mdl $src_tree_dir/tree"

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
    echo "$0: no such file $f" && exit 1;
  fi
done

if [ $stage -le -1 ]; then
  echo "$0: Prepare lang for RM-WSJ using WSJ phone set and lexicon and RM word list."
  if ! cmp -s <(grep -v "^#" $src_lang/phones.txt) <(grep -v "^#" data/lang/phones.txt); then
    local/prepare_librispeech_ami_lang.sh  $src_dict $src_lang $lang_src_tgt
  else
    rm -rf $lang_src_tgt 2>/dev/null || true
    cp -r data/lang $lang_src_tgt
  fi
  #!!! Note
  stage=4
fi

local/online/run_extractor.sh  --stage $stage \
                               --ivector-dim $ivector_dim \
                               --nnet3-affix "$nnet3_affix" \
							   --data-root $data_root \
							   --exp-root $exp_root \
                               --extractor $src_ivec_extractor_dir || exit 1;								  

train_data_dir=$data_root/${train_set}_sp_hires_comb
lores_train_data_dir=$data_root/${train_set}_sp_comb  
train_ivector_dir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb 
							   
if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $src_gmm_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    --generate-ali-from-lats true \
    $lores_train_data_dir $lang_src_tgt $src_gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz 2>/dev/null || true # save space
fi

if [ $stage -le 5 ]; then
  # Set the learning-rate-factor for all transferred layers but the last output
  # layer to primary_lr_factor.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor; set-learning-rate-factor name=output* learning-rate-factor=1.0" \
      $src_mdl $dir/input.raw || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM with librispeech and rm weight $phone_lm_scales."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=200' \
    $src_tree_dir $lat_dir $dir || exit 1;
fi

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  # exclude phone_LM and den.fst generation training stages
  if [ $train_stage -lt -4 ]; then train_stage=-4 ; fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$train_ivector_dir" \
    --chain.xent-regularize 0.1 \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch=128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=0.005 \
    --trainer.optimization.final-effective-lrate=0.0005 \
    --trainer.max-param-change 2 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $src_tree_dir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1;
fi

graph_dir=$dir/graph_poco
if [ $stage -le 18 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  # diff.
  # utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco_unk $dir $graph_dir
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_src_tgt $dir $graph_dir
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
