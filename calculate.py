from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk
import pandas as pd
from tqdm import tqdm
# nltk.download('wordnet')
import os
import time
import openai
import json
import re
import sys
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from typing import List, Union, Iterable
from itertools import zip_longest
from moverscore_v2 import word_mover_score
from collections import defaultdict
import numpy as np
import csv


def cal_UniEval(reference,answer,evaluator):
    # a list of dialogue histories
    src_list = [reference]
    # a list of additional context that should be included into the generated response
    # context_list = ['A man is using a water bottle on the street to extinguish a cigarette lit in someone else\'s hand']
    # a list of model outputs to be evaluated
    output_list = [answer]
    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, src_list=src_list)
    # Initialize evaluator for a specific task
    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data)
    return eval_scores[0]['consistency']


def mover_sentence_score(candidate: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    candidate = [candidate] * len(references)

    sentence_score = 0

    scores = word_mover_score(references, candidate, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                              remove_subwords=False)

    sentence_score = np.mean(scores)

    if trace > 0:
        print(candidate, references, sentence_score)

    return sentence_score


def cal_mover(reference, candidate):
    refs = [reference]
    mover = mover_sentence_score(candidate, refs)
    return mover

def cal_bleu4(reference, candidate):
    ref_tokens = reference.split()
    hyp_tokens = candidate.split()
    blue4_score = sentence_bleu([ref_tokens],
                                hyp_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=SmoothingFunction().method1)

    return blue4_score


def cal_rouge(reference, candidate):
    rouge = Rouge()
    rouge_score = rouge.get_scores(candidate, reference, avg=True)
    return rouge_score['rouge-l']['f']


def cal_cider(reference, candidate):
    gts = {}
    res = {}
    for idx in range(len(reference)):
        gts[idx] = [reference[idx]]
        res[idx] = [candidate[idx]]
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    return cider_score




if __name__ == '__main__':
    ref = 'A cat is sitting on the floor'
    cand = 'There is a dog running in the street'
    print(cal_mover(ref, cand))
