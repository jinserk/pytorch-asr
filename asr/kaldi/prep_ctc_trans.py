#!/usr/bin/env python

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)
# Copyright 2018       Jinserk Baik  (ICSolutions LLC)

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

# This python script converts the word-based transcripts into label sequences. The labels are
# represented by their indices.

import sys
import os.path
import glob

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: {0} <lexicon_file> <word_file> <trans_path>".format(sys.argv[0]))
        print("e.g., utils/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/train/text <UNK>")
        print("<lexicon_file> - the lexicon file in which entries have been represented by indices")
        print("<word_file>    - the word file in which entries mapped into their indices")
        print("<trans_path>   - the path contains word-based transcript files")
        exit(1)

    lexicon_file = sys.argv[1]
    word_file = sys.argv[2]
    trans_path = sys.argv[3]

    unk_word = '<unk>'
    blk_word = '<blk>'

    insert_blank = False

    # read the lexicon into a dictionary data structure
    lexicons = {}
    with open(lexicon_file, 'r') as f:
        for line in f:
            splits = line.strip().split(' ')  # assume there are no multiple spaces
            word = splits[0]
            letters = ''
            for n in range(2, len(splits)):
                letters += splits[n] + ' '
            lexicons[word] = letters.strip()
    if insert_blank:
        lexicons[blk_word] = '0'

    # read the dict file into a dictionary data structure
    words = {}
    with open(word_file, 'r') as f:
        for line in f:
            splits = line.strip().split(' ')  # assume there are no multiple spaces
            word = splits[0]
            letters = ''
            for n in range(1, len(splits)):
                letters += splits[n] + ' '
            words[word] = letters.strip()
    if insert_blank:
        words[blk_word] = blk_word

    # assume that each line is formatted as "uttid word1 word2 word3 ...", with no multiple spaces appearing
    for trans_file in glob.iglob(os.path.join(trans_path, "**/*.txt"), recursive=True):
        with open(trans_file, 'r') as rf:
            ctc_file = trans_file.replace('txt', 'ctc')
            print(f'{trans_file} -> {ctc_file}')
            with open(ctc_file, 'w') as wf:
                for line in rf:
                    out_line = ''
                    line = line.replace('\n','').strip()
                    while '  ' in line:
                        line = line.replace('  ', ' ')   # remove multiple spaces in the transcripts

                    #uttid = line.split(' ')[0]  # the first field is always utterance id
                    #trans = line.replace(uttid, '').strip()
                    trans = 'sil ' + line + ' sil'
                    #if is_char:
                    #    trans = trans.replace(' ', ' ' + blk_word + ' ')
                    splits = trans.split(' ')

                    #out_line += uttid + ' '
                    for n in range(0, len(splits)):
                        try:
                          out_line += lexicons[words[splits[n]]] + ' '
                        except Exception:
                          out_line += lexicons[words[unk_word]] + ' '
                    out_line = out_line.strip()
                    if insert_blank:
                        # insert blank symbols
                        out_line = out_line.replace(' ', ' ' + lexicons[blk_word] + ' ')
                    wf.write(out_line.strip())
