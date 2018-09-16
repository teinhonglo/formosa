#!/bin/bash
if [ $# -lt 2 ]; then
  echo "Usage: $0 <asr-results> <output-dir>"
  echo "e.g.:   local/parse_asr_result.sh \\"
  echo "    exp/tdnn/decode/ascoring/8.txt output"
  exit 1;
fi

echo "$0 $@"

file_path=$1
output_dir=$2

while IFS=: read -r line
do
    cFile=`echo $line | cut -d " " -f1 | cut -d "/" -f5-`
    cFile=`echo "${cFile/Wav/Text}"`
    content=`echo $line | cut -d " " -f2`

    fname=`basename $cFile`
    dname=$output_dir/`dirname $cFile`
    # display fields using f1, f2,..,f7
    if [ ! -d $dname ]; then
      mkdir -p $dname
    fi
    echo "$content" > $dname/${fname}.txt
    echo "$0 $fname"
    #echo "$cFile &  $content"
done <"$file_path"
