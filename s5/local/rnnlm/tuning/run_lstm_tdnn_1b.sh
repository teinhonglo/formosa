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
embedding_dim=800
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=20
lstm_rpd=200
lstm_nrpd=200
stage=0
train_stage=0
rnnlm_affix=_wsj
ac_model_dir=exp/chain/tdnn_1b_aug3
test_set=eval0
data_root=data/online
graph_affix=_13b
online_scoring=false
text=data/train_vol1_2_3b/text

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$cuda_cmd

orig_set=data/train_vol1_2_3b
wordlist=data/lang/words.txt
text_dir=data/rnnlm$rnnlm_affix
dir=exp/rnnlm_lstm_tdnn_1b$rnnlm_affix
mkdir -p $dir/config
set -e

if [ $stage -le 0 ]; then
   mkdir -p $text_dir
   echo -n >$text_dir/dev.txt
   # hold out one in every 500 lines as dev data.
   cat $text | cut -d ' ' -f1- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/formosa.txt
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
                           --top-word-features=50000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>,<SIL>' \
                           $dir/config/words.txt > $dir/config/features.txt
  
  tail -n +3 $dir/config/features.txt > $dir/config/features.txt.new
  cp $dir/config/features.txt $dir/config/features.txt.backup
  mv $dir/config/features.txt.new $dir/config/features.txt

lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1)) 
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn2 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-2))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn3 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
output-layer name=output $output_opts include-log-softmax=false dim=$embedding_dim
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
    for num_best in `seq 100 50 250`; do
      for lmwt in 0.4 ; do
        decode_dir=${ac_model_dir}/decode${graph_affix}_$test_set
        decode_dir_suffix=_${num_best}nbest${rnnlm_affix}_${lmwt}
        # Lattice rescoring
        rnnlm/lmrescore_nbest.sh \
          --cmd "$decode_cmd --mem 4G" \
          --N $num_best --skip-scoring $online_scoring $lmwt \
          data/lang${graph_affix}_test $dir \
          $data_root/${test_set}_hires ${decode_dir} \
          ${decode_dir}$decode_dir_suffix 
      
        if $online_scoring; then
          [ ! -x local/score_online.sh ] && \
            echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
            echo "score best paths"
            local/score_online.sh --cmd "$decode_cmd" $data_root/${test_set}_hires $decode_dir/../graph$graph_affix ${decode_dir}$decode_dir_suffix || exit 1
          echo "score confidence and timing with sclite"
        fi
      done	
    done
fi

exit 0
