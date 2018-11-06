#!/bin/bash

# Copyright 2012-2013  Arnab Ghoshal
#                      Johns Hopkins University (authors: Daniel Povey, Sanjeev Khudanpur)
#                      Tien-Hong Lo

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
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Script for system combination using minimum Bayes risk decoding.
# This calls lattice-combine to create a union of lattices that have been 
# normalized by removing the total forward cost from them. The resulting lattice
# is used as input to lattice-mbr-decode. This should not be put in steps/ or 
# utils/ since the scores on the combined lattice must not be scaled.
set -euo pipefail
# begin configuration section.
cmd=run.pl
beam=4 # prune the lattices prior to MBR decoding, for speed.
stage=0
cer=0
decode_mbr=true
lat_weights=
word_ins_penalty=0.0
min_lmwt=5
max_lmwt=15
frame_shift_opts=0.03
parallel_opts="--num-threads 3"
skip_scoring=false
ctm_name=
overlap_spk=4
asclite=false
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <data-dir> <graph-dir|lang-dir> <decode-dir1>[:lmwt-bias] <decode-dir2>[:lmwt-bias] [<decode-dir3>[:lmwt-bias] ... ] <out-dir>
     E.g. "$(basename $0)" data/test data/lang exp/tri1/decode exp/tri2/decode exp/tri3/decode exp/combine
     or:  "$(basename $0)" data/test data/lang exp/tri1/decode exp/tri2/decode:18 exp/tri3/decode:13 exp/combine
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --min-lmwt INT                  # minumum LM-weight for lattice rescoring 
  --max-lmwt INT                  # maximum LM-weight for lattice rescoring
  --lat-weights STR               # colon-separated string of lattice weights
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --stage (0|1|2)                 # (createCTM | filterCTM | runSclite).
  --parallel-opts <string>        # extra options to command for combination stage,
                                  # default '--num-threads 3'
  --cer (0|1)                     # compute CER in addition to WER
";

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 5 ]; then
  printf "$help_message\n";
  exit 1;
fi

echo "$0 $@"
data=$1
lang=$2
dir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

#Let the user to set the CTM file name 
#use the data-dir name in case the user doesn't care
if [ -z ${ctm_name} ] ; then
  ctm_name=`basename $data`
fi

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $lang/words.txt $lang/phones/word_boundary.int ; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done
if ! $skip_scoring ; then
  for f in  $data/stm; do
    [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
  done
fi


mkdir -p $dir/log

for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  decode_dir=`echo $decode_dir | cut -d: -f1`
  [ -z "$offset" ] && offset=0
  
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
    lattice-limit-depth ark:- ark:- | \
    lattice-push --push-strings=false ark:- ark:- | \
    lattice-align-words-lexicon --max-expand=10.0 \
      $lang/phones/align_lexicon.int $model ark:- ark:- |"
done

mkdir -p $dir/ascoring/log

if [ -z "$lat_weights" ]; then
  lat_weights=1.0
  for i in `seq $[$num_sys-1]`; do lat_weights="$lat_weights:1.0"; done
fi

if [ $stage -le 0 ]; then  
  echo "$0 lattice combination"
  $cmd $parallel_opts LMWT=$min_lmwt:$max_lmwt $dir/log/combine_lats.LMWT.log \
    mkdir -p $dir/ascore_LMWT/ '&&' \
    lattice-combine --lat-weights=$lat_weights "${lats[@]}" ark:- \| \
    lattice-to-ctm-conf --decode-mbr=true ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt  \| \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
    '>' $dir/ascore_LMWT/${ctm_name}.ctm || exit 1;
fi


if [ $stage -le 1 ]; then
# Remove some stuff we don't want to score, from the ctm.
# - we remove hesitations here, otherwise the CTM would have a bug!
#   (confidences in place of the removed hesitations),
  for lmwt in `seq $min_lmwt $max_lmwt`; do
    x=$dir/ascore_${lmwt}/${ctm_name}.ctm
    [ ! -f $x ] && echo "File $x does not exist! Exiting... " && exit 1
    cp $x $x.bkup1;
    cat $x.bkup1 | grep -i -v -E '\[noise|laughter|vocalized-noise\]' | \
      grep -i -v -E ' (ACH|AH|EEE|EH|ER|EW|HA|HEE|HM|HMM|HUH|MM|OOF|UH|UM) ' | \
      grep -i -v -E '<unk>' > $x;
    cp $x $x.bkup2;
  done
fi

if ! $skip_scoring ; then
  if [ $stage -le 2 ]; then
    if [ "$asclite" == "true" ]; then
      oname=$ctm_name
	  echo $ctm_name
      [ ! -z $overlap_spk ] && oname=${ctm_name}_o$overlap_spk
      echo "asclite is starting"
      # Run scoring, meaning of hubscr.pl options:
      # -G .. produce alignment graphs,
      # -v .. verbose,
      # -m .. max-memory in GBs,
      # -o .. max N of overlapping speakers,
      # -a .. use asclite,
      # -C .. compression for asclite,
      # -B .. blocksize for asclite (kBs?),
      # -p .. path for other components,
      # -V .. skip validation of input transcripts,
      # -h rt-stt .. removes non-lexical items from CTM,
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
      cp $data/stm $dir/ascore_LMWT/ '&&' \
      cp $dir/ascore_LMWT/${ctm_name}.ctm $dir/ascore_LMWT/${oname}.ctm '&&' \
      $hubscr -G -v -m 1:2 -o$overlap_spk -a -C -B 8192 -p $hubdir -V -l english \
        -h rt-stt -g $data/glm -r $dir/ascore_LMWT/stm $dir/ascore_LMWT/${oname}.ctm || exit 1
      # Compress some scoring outputs : alignment info and graphs,
      echo -n "compressing asclite outputs "
      for LMWT in $(seq $min_lmwt $max_lmwt); do
        ascore=$dir/ascore_${LMWT}
        gzip -f $ascore/${oname}.ctm.filt.aligninfo.csv
        cp $ascore/${oname}.ctm.filt.alignments/index.html $ascore/${oname}.ctm.filt.overlap.html
        tar -C $ascore -czf $ascore/${oname}.ctm.filt.alignments.tar.gz ${oname}.ctm.filt.alignments
        rm -r $ascore/${oname}.ctm.filt.alignments
        echo -n "LMWT:$LMWT "
       done
       echo done
    else
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
        cp $data/stm $dir/ascore_LMWT/ '&&' \
        $hubscr -p $hubdir -v -V -l english -h hub5 -g $data/glm -r $dir/ascore_LMWT/stm $dir/ascore_LMWT/${ctm_name}.ctm || exit 1
    fi
  fi
fi


exit 0
