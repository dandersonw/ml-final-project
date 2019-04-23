# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --crosslingual --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec

import os
import argparse
import csv
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp
from src.models import build_model, update_src_embeddings
from src.trainer import Trainer
from src.evaluation import Evaluator

# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="ja", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _, srcWord2FreqRank = build_model(params, False, 1)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
evaluator = Evaluator(trainer)

# run evaluations
to_log = OrderedDict({'n_iter': 0})
if params.tgt_lang:

    emb_idx = 1
    while True:
        src_word_ids, matches = evaluator.word_translation(to_log)

        with open('matches_500k_4500_batch.csv', 'a') as fp:
            wr = csv.writer(fp, delimiter=',')
            for idx, m in enumerate(matches):
                word = evaluator.src_dico.id2word[src_word_ids[idx]]
    
                line = []
                line.append(word)
                for word_id in m:
                    line.append(evaluator.tgt_dico.id2word[word_id])
                line.append(srcWord2FreqRank[word])
    
                wr.writerow(line)

        del trainer
        del evaluator

        emb_idx += 4500

        if emb_idx > 200312:
            break

        srcWord2FreqRank = update_src_embeddings(src_emb, params, emb_idx)
        trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
        evaluator = Evaluator(trainer)
