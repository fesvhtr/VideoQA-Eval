import sys
sys.path.append('/mnt/new_disk/dh/zsc/data/VideoQA-Eval/UniEval')


import nltk
import pandas as pd
import argparse
from tqdm import tqdm
# nltk.download('wordnet')
import os
import time
import openai
import json
import re
import sys
from typing import List, Union, Iterable
from itertools import zip_longest
from collections import defaultdict
import numpy as np
import csv

from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
# from moverscore_v2 import word_mover_score


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

def sentence_score(hypothesis: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    hypothesis = [hypothesis] * len(references)

    sentence_score = 0

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                              remove_subwords=False)

    sentence_score = np.mean(scores)

    if trace > 0:
        print(hypothesis, references, sentence_score)

    return sentence_score

def cal_mover(reference, answer):
    refs = [reference]
    ans = answer
    mover = sentence_score(ans, refs)
    return mover

def cal_bleu4(reference, hypothesis):
    # 将字符串分词为列表
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    blue4_score = sentence_bleu([ref_tokens],
                                hyp_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=SmoothingFunction().method1)

    return blue4_score

def cal_rouge(reference, hypothesis):
    rouge = Rouge()
    rouge_score = rouge.get_scores(hypothesis, reference, avg=True)
    return rouge_score['rouge-l']['f']

def cal_cider(reference, candidate):
    gts = {}
    res = {}
    for idx in range(len(reference)):
        gts[idx] = [reference[idx]]
        res[idx] = [candidate[idx]]

    # 初始化CIDEr评估器
    cider_scorer = Cider()

    # 计算CIDEr分数
    cider_score, _ = cider_scorer.compute_score(gts, res)
    return cider_score

def cal_bleurt(gt,sub):
    from bleurt import score

    checkpoint = "../BLEURT-20"
    references = [gt]
    candidates = [sub]

    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    assert isinstance(scores, list) and len(scores) == 1
    return scores

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for video captions.')
    parser.add_argument('--sub', type=str, help='Subtitle text', default='A man is seen walking away from a car.')
    parser.add_argument('--gt', type=str, help='Ground truth text', default='Outside the market, a thief is taking something out of a car with a shattered window.')
    parser.add_argument('--metric', nargs='+', type=str, help='Metrics to calculate', choices=['bleu', 'rouge', 'moverscore', 'bleurt', 'unieval', 'all'], default=['all'])
    args = parser.parse_args()

    metrics_to_run = args.metric if 'all' not in args.metric else ['bleu4', 'rouge', 'moverscore', 'bleurt', 'unieval']

    for metric in metrics_to_run:
        if metric == 'bleu':
            result = cal_bleu4(args.gt, args.sub)
        elif metric == 'rouge':
            result = cal_rouge(args.gt, args.sub)
        elif metric == 'moverscore':
            result = cal_mover(args.gt, args.sub)
        elif metric == 'bleurt':
            result = cal_bleurt(args.gt, args.sub)
        elif metric == 'unieval':
            evaluator = get_evaluator('fact')
            result = cal_UniEval(args.gt, args.sub, evaluator)
        else:
            print(f'Invalid metric "{metric}" specified')
            continue

        print(f'{metric.upper()}: {result}')

if __name__ == '__main__':
    main()


