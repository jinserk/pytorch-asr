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
import argparse
from pathlib import Path

class PrepareCtc:
    unk_word = '<unk>'
    blk_word = '<blk>'

    def __init__(self, lexicon_file, label_file, insert_blank=False):
        self.insert_blank = insert_blank

        self.lexicons = {}
        self.labels = {}

        self._load_lexicon_file(args.lexicon_file)
        self._load_label_file(args.label_file)

        # for counting priors per label
        self.label_counts = [0] * len(self.labels)

    def _load_lexicon_file(self, lexicon_file):
        # read the lexicon into a dictionary data structure
        with open(lexicon_file, 'r') as f:
            for line in f:
                splits = line.strip().split()
                word = splits[0]
                self.lexicons[word] = splits[2:]

    def _load_label_file(self, label_file):
        # read the label file into a dictionary data structure
        with open(label_file, 'r') as f:
            for line in f:
                splits = line.strip().split()
                label = splits[0]
                self.labels[label] = splits[1]
        if self.insert_blank and self.blk_word not in self.labels:
            self.labels[self.blk_word] = 1

    def convert(self, trans_path):
        # assume that each line is formatted as "word1 word2 word3 ...", with no multiple spaces appearing
        for trans_file in Path(trans_path).rglob("*.txt"):
            with open(trans_file, 'r') as rf:
                ctc_file = str(trans_file).replace('txt', 'ctc')
                print(f"{trans_file} -> {ctc_file}")
                with open(ctc_file, 'w') as wf:
                    for line in rf:
                        trans = line.strip().split()

                        out = list()
                        for w in trans:
                            try:
                                for t in self.lexicons[w]:
                                    out.append(self.labels[t])
                            except Exception:
                                out.extend([self.labels[t] for t in self.lexicons[self.unk_word]])

                        for c in out:   # label counts for priors
                            self.label_counts[int(c)] += 1

                        out_line = ' '.join(out)
                        if self.insert_blank:
                            # insert blank symbols
                            out_line = out_line.replace(' ', ' ' + self.labels[self.blk_word] + ' ')

                        wf.write(out_line.strip())

    def write_label_counts(self, count_file):
        print(self.label_counts)
        with open(count_file, 'w') as f:
            counts = [str(c) for c in self.label_counts]
            f.write(' '.join(counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare CTC labels")
    parser.add_argument('--lexicon-file', type=str, default='graph/align_lexicon.txt', help="the lexicon file in which entries have been represented by labels")
    parser.add_argument('--label-file', type=str, default='graph/labels.txt', help="the label file in which entries mapped into their label indices")
    parser.add_argument('--count-file', type=str, default='label_counts.txt', help="output file for occurence count of each label to calculate priors")
    parser.add_argument('trans_paths', type=str, nargs='+', help="list of paths containing transcript txt files to be converted")
    args = parser.parse_args()

    h = PrepareCtc(lexicon_file=args.lexicon_file, label_file=args.label_file)

    for trans_path in args.trans_paths:
        h.convert(trans_path)

    count_file = Path(args.trans_paths[0]).joinpath(args.count_file)
    h.write_label_counts(count_file)
