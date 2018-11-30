#!/bin/bash
text="data/local/train/text_2_3b"
word_seg="data/local/train/word_seg_vocab.txt"
for c in `awk '{print $1}' $word_seg`; do
  num=`grep -o -w "$c" $text | wc -l`
  echo "$c $((99 + $num))"
done
exit 0;
