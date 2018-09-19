#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
data_root=data/matbn
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

function filter_text {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; } 
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}) { print "$a "; }} print "\n"; }' \
   '[NOISE]' '[LAUGHTER]' '[VOCALIZED-NOISE]' '<UNK>' '<unk>' '%HESITATION'' {french}' '{hindi}' '{japanese}' '{laugh}' '{noise}' '{sanskrit}' '{sil}' '<SIL>' '{sing}' '{taiyu}' '<sil>' '[A]' '[NEI]' '[S]' '<BREATHE>' '[NA]' '[HOU]' '<ENGLISH>' '[E]' '<SPOKEN_NOISE>' '[LA]' '[EI]' '[MA]' '[NE]' '[EN]' '[O]' '<FORMOSAN>' '[CHII]' '.' '！' '<HAKKA>' '<MIN-NAN>' '[NEI]' '[BA]' '[S]' '[HA]' '[NO]' '[AI]' '[IOU]' '[TZ]' '[MAI]' '<FRENCH>' '<UNKNOWN>' '[HM]' '[TS]' '[SHII]' '：' '[IU]' '[EIN]' '[HEI]' '．' '[CHU]' '[TAU]' '[JEN]' '<TURKISH>' '[LO]' '[MI]' '[IA]'
}

for datadir in matbn_dev matbn_eval; do
  filter_text <$data_root/$datadir/text > $data_root/$datadir/text.filt
  python local/extract.py $data_root/$datadir/text.filt > $data_root/$datadir/text.filt.clean
  awk '{print $1}' $data_root/$datadir/text.filt.clean > $data_root/$datadir/uttlist
  utils/data/subset_data_dir.sh --utt-list $data_root/$datadir/uttlist $data_root/$datadir $data_root/${datadir}_cleanup
  rm $data_root/$datadir/text.filt $data_root/$datadir/text.filt.clean $data_root/$datadir/uttlist
done
echo "Done"
exit 0
