#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# This script is similar to rnnlm_lstm_tdnn_a.sh except for adding L2 regularization.

# rnnlm/train_rnnlm.sh: best iteration (out of 18) was 17, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 45.6 / 68.7.
# Train objf: -651.50 -4.44 -4.26 -4.15 -4.08 -4.03 -4.00 -3.97 -3.94 -3.92 -3.90 -3.89 -3.88 -3.86 -3.85 -3.84 -3.83 -3.82
# Dev objf:   -10.76 -4.68 -4.47 -4.38 -4.33 -4.29 -4.28 -4.27 -4.26 -4.26 -4.25 -4.24 -4.24 -4.24 -4.23 -4.23 -4.23 -4.23

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_1a
embedding_dim=800
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=20
lstm_rpd=200
lstm_nrpd=200
stage=0
train_stage=0

ac_model_dir=exp/chain/tdnn_1b_aug3
test_set=test
data_root=data

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$cuda_cmd

train=data/train_vol1_2_3_sub90/text
dev=data/dev_sub10/text
wordlist=data/lang/words.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

if [ ! -f $dev ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 data/train_vol1_2_3  data/train_vol1_2_3_sub90 data/dev_sub10
fi

for f in $train $dev $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for run.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $train | cut -d ' ' -f2- > $text_dir/formosa.txt
  cat $dev | cut -d ' ' -f2- > $text_dir/dev.txt
fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
formosa  1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features 50000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>,#0' \
                           $dir/config/words.txt > $dir/config/features.txt
  tail -n +3 $dir/config/features.txt > $dir/config/features.txt.new
  cp $dir/config/features.txt $dir/config/features.txt.backup
  mv $dir/config/features.txt.new $dir/config/features.txt

lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-3))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-3))
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --embedding_l2 $embedding_l2 \
                       --stage $train_stage \
                       --num-egs-threads 2 \
                       --num-epochs $epochs --cmd "$cmd" $dir
fi

if [ $stage -le 4 ]; then
    decode_dir=${ac_model_dir}/decode_${graph_affix}_test
    decode_dir_suffix=_nbest
    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.8 --N 20 \
      data/lang${graph_affix}_${test_set} $dir \
      $data_root/${test_set}_hires ${decode_dir} \
      ${decode_dir}${graph_affix}_${test_set} &
  wait
fi

exit 0
