#!/bin/bash

# Copyright 2012-2013  Arnab Ghoshal
#                      Johns Hopkins University (authors: Daniel Povey, Sanjeev Khudanpur)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific lang_or_graphuage governing permissions and
# limitations under the License.


# Script for system combination using minimum Bayes risk decoding.
# This calls lattice-combine to create a union of lattices that have been 
# normalized by removing the total forward cost from them. The resulting lattice
# is used as input to lattice-mbr-decode. This should not be put in steps/ or 
# utils/ since the scores on the combined lattice must not be scaled.

# begin configuration section.
cmd=queue.pl
beam=8 # prune the lattices prior to MBR decoding, for speed.
word_ins_penalty=0.5
stage=0
cer=0
decode_mbr=true
lat_weights=
min_lmwt=5
max_lmwt=15
parallel_opts="--num-threads 1"
skip_scoring=false
ctm_name=
decode_mbr=true
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <data-dir> <graph-dir|lang_or_graph-dir> <decode-dir1>[:lmwt-bias] <decode-dir2>[:lmwt-bias] [<decode-dir3>[:lmwt-bias] ... ] <out-dir>
     E.g. "$(basename $0)" data/test data/lang_or_graph exp/tri1/decode exp/tri2/decode exp/tri3/decode exp/combine
     or:  "$(basename $0)" data/test data/lang_or_graph exp/tri1/decode exp/tri2/decode:18 exp/tri3/decode:13 exp/combine
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --min-lmwt INT                  # minumum LM-weight for lattice rescoring 
  --max-lmwt INT                  # maximum LM-weight for lattice rescoring
  --lat-weights STR               # colon-separated string of lattice weights
  --stage (0|1|2)                 # (createCTM | filterCTM | runSclite).
  --parallel-opts <string>        # extra options to command for combination stage,
                                  # default '--num-threads 3'
  --cer (0|1)                     # compute CER in addition to WER
";

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

export LC_ALL=en_US.UTF-8

if [ $# -lt 5 ]; then
  printf "$help_message\n";
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine


ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"

for f in $lang_or_graph/words.txt; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done

symtab=$lang_or_graph/words.txt

mkdir -p $dir/log

# echo "$0 Appending lattice commands"
for i in `seq 0 $[num_sys-1]`; do
  # echo "Doing Model $i"
  decode_dir=${decode_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  decode_dir=`echo $decode_dir | cut -d: -f1`
  [ -z "$offset" ] && offset=0
  echo "$decode_dir " >> $dir/decode_dir.txt    
  model=`dirname $decode_dir`/final.mdl  # model one level up from decode dir
  for f in $model $decode_dir/lat.1.gz ; do
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
  done
  if [ $i -eq 0 ]; then
    nj=`cat $decode_dir/num_jobs` || exit 1;
  else
    if [ $nj != `cat $decode_dir/num_jobs` ]; then
      echo "$0: number of decoding jobs mismatches, $nj versus `cat $decode_dir/num_jobs`" 
      exit 1;
    fi
  fi
  file_list=""
  # I want to get the files in the correct order so we can use ",s,cs" to avoid
  # memory blowup.  I first tried a pattern like file.{1,2,3,4}.gz, but if the
  # system default shell is not bash (e.g. dash, in debian) this will not work,
  # so we enumerate all the input files.  This tends to make the command lines
  # very long.
  for j in `seq $nj`; do file_list="$file_list $decode_dir/lat.$j.gz"; done

  lats[$i]="ark,s,cs:lattice-scale --inv-acoustic-scale=\$[$offset+LMWT] 'ark:gunzip -c $file_list|' ark:- | \
     lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- | \
     lattice-prune --beam=$beam ark:- ark:- |"
done

mkdir -p $dir/scoring/log

# echo "$0 combine to lattice"
if [ -z "$lat_weights" ]; then
    lat_weights=1.0
    for i in `seq $[$num_sys-1]`; do lat_weights="$lat_weights:1.0"; done
fi


echo "$0 lat_weights=$lat_weights, decode-mbr=$decode_mbr"
mkdir -p $dir/scoring/penalty_$word_ins_penalty
$cmd $parallel_opts LMWT=$min_lmwt:$max_lmwt $dir/log/combine_lats.LMWT.log \
    lattice-combine --lat-weights=$lat_weights "${lats[@]}" ark:- \| \
    lattice-mbr-decode --word-symbol-table=$symtab ark:- ark,t:- \| \
    utils/int2sym.pl -f 2- $symtab \| \
    $hyp_filtering_cmd '>' $dir/scoring/penalty_$word_ins_penalty/LMWT.txt || exit 1;

cp $decode_dir/scoring_kaldi/test_filt.txt $dir/scoring/test_filt.txt
cp $decode_dir/scoring_kaldi/test_filt.chars.txt $dir/scoring/test_filt.chars.txt
	
if [ $stage -le 2 ] ; then
  files=($decode_dir/scoring_kaldi/test_filt.txt)
  for wip in $(echo $word_ins_penalty); do
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      files+=($dir/scoring/penalty_${wip}/${lmwt}.txt)
    done
  done

  for f in "${files[@]}" ; do
    fout=${f%.txt}.chars.txt
    if [ -x local/character_tokenizer ]; then
      cat $f |  local/character_tokenizer > $fout
    else
      cat $f |  perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $fout
    fi
  done

  for wip in $(echo $word_ins_penalty); do
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/penalty_$wip/log/score.cer.LMWT.log \
      cat $dir/scoring/penalty_$wip/LMWT.chars.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring/test_filt.chars.txt  ark,p:- ">&" $dir/cer_LMWT_$wip || exit 1;
  done
fi

if [ $stage -le 3 ] ; then
  for wip in $(echo $word_ins_penalty); do
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      # adding /dev/null to the command list below forces grep to output the filename
      grep WER $dir/cer_${lmwt}_${wip} /dev/null
    done
  done | utils/best_wer.sh  >& $dir/scoring/best_cer || exit 1

  best_cer_file=$(awk '{print $NF}' $dir/scoring/best_cer)
  best_wip=$(echo $best_cer_file | awk -F_ '{print $NF}')
  best_lmwt=$(echo $best_cer_file | awk -F_ '{N=NF-1; print $N}')

  if [ -z "$best_lmwt" ]; then
    echo "$0: we could not get the details of the best CER from the file $dir/cer_*.  Probably something went wrong."
    exit 1;
  fi

  if $stats; then
    mkdir -p $dir/scoring/cer_details
    echo $best_lmwt > $dir/scoring/cer_details/lmwt # record best lang_or_graphuage model weight
    echo $best_wip > $dir/scoring/cer_details/wip # record best word insertion penalty

    $cmd $dir/scoring/log/stats1.cer.log \
      cat $dir/scoring/penalty_$best_wip/${best_lmwt}.chars.txt \| \
      align-text --special-symbol="'***'" ark:$dir/scoring/test_filt.chars.txt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring/cer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring/cer_details/per_spk || exit 1;

    $cmd $dir/scoring/log/stats2.cer.log \
      cat $dir/scoring/cer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring/cer_details/ops || exit 1;

    $cmd $dir/scoring/log/cer_bootci.cer.log \
      compute-wer-bootci --mode=present \
        ark:$dir/scoring/test_filt.chars.txt ark:$dir/scoring/penalty_$best_wip/${best_lmwt}.chars.txt \
        '>' $dir/scoring/cer_details/cer_bootci || exit 1;

  fi
fi
