#!/bin/bash

# setup envs
if [ $# -ne 1 ]; then
  echo "KALDI_ROOT should be given as an argument"
  exit 1;
fi

CWD=$(dirname $0 | xargs readlink -f)
KALDI_ROOT=$1
export PATH=$KALDI_ROOT/tools/openfst/bin:$CWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

model_dir=$(readlink -f "$CWD/aspire")
lang_dir=$(readlink -f "$model_dir/data/lang_pp_test")
out_dir=$(readlink -f "$CWD/../graph")

# aspire model directory check
if [ ! -e $lang_dir ]; then
  mkdir -p $model_dir; cd $model_dir
  if [ ! -e ./0001_aspire_chain_model.tar.gz ]; then
    echo "downloading pretrained ASpIRE model"
    wget --no-check-certificate --quiet http://dl.kaldi-asr.org/models/0001_aspire_chain_model.tar.gz
    tar zxf 0001_aspire_chain_model.tar.gz
  else
    echo "model file already downloaded, but not unzipped"
    tar -zxf 0001_aspire_chain_model.tar.gz
  fi
  cd ..
else
  echo "model directory already exists, skipping downloading"
fi

# move files required for training
mkdir -p $out_dir
cp $lang_dir/words.txt $out_dir
cp $lang_dir/phones.txt $out_dir
cp $lang_dir/phones/align_lexicon.int $out_dir

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (e.g.,
# phonemes), and the disambiguation symbols.  
phn_dir=$lang_dir/phones
(echo '<blk>';) | cat - $phn_dir/silence.txt $phn_dir/nonsilence.txt | \
  awk '{print $1 " " (NR-1)}' > $out_dir/labels.txt
(echo '<eps>'; echo '<blk>';) | cat - $phn_dir/silence.txt $phn_dir/nonsilence.txt $phn_dir/disambig.txt | \
  awk '{print $1 " " (NR-1)}' > $out_dir/tokens.txt

# Compile the tokens into FST
t_fst=$out_dir/T.fst
t_tmp=$t_fst.$$
trap "rm -f $t_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $t_fst ]]; then
  $CWD/ctc_token_fst.py $out_dir/tokens.txt | \
    fstcompile --isymbols=$out_dir/tokens.txt --osymbols=$out_dir/phones.txt \
    --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $t_tmp || exit 1;
  mv $t_tmp $t_fst
  echo "Composing decoding graph T.fst succeeded"
else
  echo "T.fst already exists and is new"
fi
    
# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and minimized.
lg_fst=$out_dir/LG.fst
lg_tmp=$lg_fst.$$
trap "rm -f $lg_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $lg_fst || $lg_fst -ot $lang_dir/G.fst || $lg_fst -ot $lang_dir/L_disambig.fst ]]; then
  fsttablecompose $lang_dir/L_disambig.fst $lang_dir/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstpushspecial | fstarcsort --sort_type=ilabel > $lg_tmp || exit 1;
  mv $lg_tmp $lg_fst
  fstisstochastic $lg_fst || echo "[info]: $lg_fst is not stochastic"
  echo "Composing decoding graph LG.fst succeeded"
else
  echo "LG.fst already exists and is new"
fi

if false; then
  N=3 #$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
  P=1 #$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }
  clg_fst=$out_dir/CLG.fst
  clg_tmp=$clg_fst.$$
  ilabels=$out_dir/ilabels_$N_$P
  ilabels_tmp=$ilabels.$$
  trap "rm -f $clg_tmp $ilabels_tmp" EXIT HUP INT PIPE TERM
  if [[ ! -s $clg_fst || $clg_fst -ot $lg_fst || ! -s $ilabels || $ilabels -ot $lg_fst ]]; then
    fstcomposecontext --context-size=$N --central-position=$P \
     --read-disambig-syms=$lang_dir/phones/disambig.int \
     --write-disambig-syms=$out_dir/disambig_ilabels_${N}_${P}.int \
      $ilabels_tmp < $lg_fst | \
      fstarcsort --sort_type=ilabel > $clg_tmp
    mv $clg_tmp $clg_fst
    mv $ilabels_tmp $ilabels
    fstisstochastic $clg_fst || echo "[info]: $clg_fst is not stochastic."
  else
    echo "CLG.fst already exists and is new"
  fi

  #fsttablecompose $t_fst $clg_fst > $out_dir/TCLG.fst || exit 1;
  #echo "Composing decoding graph TCLG.fst succeeded"
fi

tlg_fst=$out_dir/TLG.fst

trap "rm -f $tlg_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $tlg_fst || $tlg_fst -ot $t_fst || $tlg_fst -ot $lg_fst ]]; then
  fsttablecompose $t_fst $lg_fst > $tlg_tmp || exit 1;
  mv $tlg_tmp $tlg_fst
  echo "Composing decoding graph TLG.fst succeeded"
else
  echo "TLG.fst already exists and is new"
fi

