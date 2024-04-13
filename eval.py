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


def calculate_UniEval(reference,answer,evaluator):
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


def calculate_mover(reference, answer):
    refs = [reference]
    ans = answer
    mover = sentence_score(ans, refs)
    return mover

def calculate_bleu4(reference, hypothesis):
    # 将字符串分词为列表
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    blue4_score = sentence_bleu([ref_tokens],
                                hyp_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=SmoothingFunction().method1)

    return blue4_score


def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    rouge_score = rouge.get_scores(hypothesis, reference, avg=True)
    return rouge_score['rouge-l']['f']


def compute_cider_score(reference, candidate):
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

def chk_file(submission_file, answer_file):
    with open(submission_file) as f:
        submission = json.load(f)
    with open(answer_file) as f:
        answer = json.load(f)

    chk_answer = []
    for data in answer:
        chk_answer.append({
            'task': data['task'],
            'visual_input': data['visual_input'],
            'ID': data['ID']
        })

    diff = False
    for data in submission:
        if {
            'task': data['task'],
            'visual_input': data['visual_input'],
            'ID': data['ID']
        } not in chk_answer:
            print(data)
            diff = True
            break

    assert not diff, 'Submission file is not valid'
    print('File is valid! Loading File...')

    submission = sorted(submission, key=lambda x: x['ID'])
    answer = sorted(answer, key=lambda x: x['ID'])

    # 假设 submission 和 answer 已经是排序过的列表
    submission = sorted(submission, key=lambda x: x['ID'])
    answer = sorted(answer, key=lambda x: x['ID'])

    # 创建一个集合来存储相同的 ID 和 output
    duplicate_id_output = set()

    # 找出 submission 和 answer 中相同的 ID 和 output
    for sub_data in submission:
        for ans_data in answer:
            if sub_data['ID'] == ans_data['ID'] and sub_data['output'] == ans_data['output']:
                duplicate_id_output.add((sub_data['ID'], sub_data['output']))

    # 过滤掉 submission 和 answer 中存在于 duplicate_id_output 的项
    filtered_submission = [data for data in submission if (data['ID'], data['output']) not in duplicate_id_output]
    filtered_answer = [data for data in answer if (data['ID'], data['output']) not in duplicate_id_output]

    # 更新 submission 和 answer 列表
    submission = filtered_submission
    answer = filtered_answer
    submission = sorted(submission, key=lambda x: x['ID'])
    answer = sorted(answer, key=lambda x: x['ID'])
    print('submission: ',len(submission),'answer: ',len(answer))
    return submission,answer

def eval(submission_file, answer_file, total_score_path, run_time):
    print('Validating...')
    col = ['Task', 'Cause', 'Result', 'Description']

    submission,answer = chk_file(submission_file, answer_file)

    can_path = 'bleurt/test_data/candidates'
    ref_path = 'bleurt/test_data/references'
    with open(can_path, 'w') as f_can:
        f_can.close()
    with open(ref_path, 'w') as f_ref:
        f_ref.close()

    bleurt_score_path = 'score.txt'
    eval_csv = pd.DataFrame(
        columns=['pre_output', 'gt', 'Task', 'bleurt_score'])
    for i in tqdm(range(len(submission))):
        if submission[i]['task'] in ['Detection', 'Timestamp', 'Classification']:
            continue
        pre_output = submission[i]['output'].replace('\n', ' ')
        gt = answer[i]['output'].replace('\n', ' ')

        with open(can_path, 'a') as f_can:
            f_can.write(pre_output + '\n')
        with open(ref_path, 'a') as f_ref:
            f_ref.write(gt + '\n')
        task = answer[i]['task']
        eval_csv = pd.concat([
            eval_csv,
            pd.DataFrame([[pre_output, gt, task, 0]],
                         columns=['pre_output', 'gt', 'Task', 'bleurt_score'])
        ])

    f_can.close()
    f_ref.close()

    os.system(
        'python -m bleurt.score_files   -candidate_file={}  -reference_file={}   -bleurt_checkpoint=BLEURT-20   -scores_file={}'
        .format(can_path, ref_path, bleurt_score_path))

    from time import sleep
    sleep(run_time)

    with open(bleurt_score_path) as f:
        eval_csv['bleurt_score'] = [i[:-1] for i in f.readlines()]

    rouge_score, bleu_score, bleurt_score, cider_score = {}, {}, {}, {}
    bleurt_score['Cause'], bleurt_score['Result'], bleurt_score['Description'] = [], [], []
    rouge_score['Cause'], rouge_score['Result'], rouge_score['Description'] = [], [], []
    bleu_score['Cause'], bleu_score['Result'], bleu_score['Description'] = [], [], []

    for index, row in eval_csv.iterrows():

        if row['pre_output'] == '':
            bleu_s = 0.
            rouge_s = 0.
            bleurt_s = 0.
        else:
            groudtruth_value = str(row['gt'])
            groudtruth_value.lower()
            result_value = str(row['pre_output'])
            result_value.lower()

            bleu_s = calculate_bleu4(groudtruth_value, result_value)
            rouge_s = calculate_rouge(groudtruth_value, result_value)
            bleurt_s = float(row['bleurt_score'])

        if row['Task'] == 'Cause':
            bleu_score['Cause'].append(bleu_s)
            rouge_score['Cause'].append(rouge_s)
            bleurt_score['Cause'].append(bleurt_s)
        elif row['Task'] == 'Result':
            bleu_score['Result'].append(bleu_s)
            rouge_score['Result'].append(rouge_s)
            bleurt_score['Result'].append(bleurt_s)
        elif row['Task'] == 'Description':
            bleu_score['Description'].append(bleu_s)
            rouge_score['Description'].append(rouge_s)
            bleurt_score['Description'].append(bleurt_s)

    bleu_score_ls, rouge_score_ls, bleurt_score_ls, cider_score_ls, weighted_avg_ls = [], [], [], [], []

    for task in col[1:]:
        df = eval_csv[eval_csv['Task'] == task]
        cider_score_ls.append(
            compute_cider_score(list(df['gt']), list(df['pre_output'])) * 10)
        bleu_score_ls.append(
            sum(bleu_score[task]) / len(bleu_score[task]) * 100)
        rouge_score_ls.append(
            sum(rouge_score[task]) / len(rouge_score[task]) * 100)
        bleurt_score_ls.append(
            sum(bleurt_score[task]) / len(bleurt_score[task]) * 100)
        weighted_avg_ls.append(bleu_score_ls[-1] * 0.2 +
                               rouge_score_ls[-1] * 0.2 +
                               cider_score_ls[-1] * 0.2 +
                               bleurt_score_ls[-1] * 0.4)

    score_df = pd.DataFrame(columns=['Eval'] + col[1:])
    score_df.loc[len(score_df)] = ['BLEU-4'] + bleu_score_ls
    score_df.loc[len(score_df)] = ['ROUGE-L'] + rouge_score_ls
    score_df.loc[len(score_df)] = ['BLEURT'] + bleurt_score_ls
    score_df.loc[len(score_df)] = ['CIDEr'] + cider_score_ls
    score_df.loc[len(score_df)] = ['Weighted Avg'] + weighted_avg_ls
    score_df.to_csv(total_score_path, index=False)

    print('Classic Evaluation Finish! Score File was save in ' +
          total_score_path)


def eval_v2(submission_file, answer_file, total_score_path):
    print('Validating...')
    col = ['Task', 'Cause', 'Result', 'Description']
    submission, answer = chk_file(submission_file, answer_file)

    unieval_score, mover_score = {}, {}
    unieval_score['Cause'], unieval_score['Result'], unieval_score['Description'] = [], [], []
    mover_score['Cause'], mover_score['Result'], mover_score['Description'] = [], [], []

    evaluator = get_evaluator('fact')
    for i in tqdm(range(len(submission))):
        if submission[i]['ID'] != answer[i]['ID']:
            print('ID not match!')
        if submission[i]['task'] in ['Detection', 'Timestamp', 'Classification']:
            continue

        pre_output = submission[i]['output'].replace('\n', ' ')
        gt = answer[i]['output'].replace('\n', ' ')
        try:
            if submission[i]['task'] == 'Cause':
                unieval_score['Cause'].append(calculate_UniEval(gt,pre_output,evaluator))
                mover_score['Cause'].append(calculate_mover(reference=gt, answer=pre_output))

            elif submission[i]['task'] == 'Result':
                unieval_score['Result'].append(calculate_UniEval(gt,pre_output,evaluator))
                mover_score['Result'].append(calculate_mover(reference=gt, answer=pre_output))

            elif submission[i]['task'] == 'Description':
                unieval_score['Description'].append(calculate_UniEval(gt,pre_output,evaluator))
                mover_score['Description'].append(calculate_mover(reference=gt, answer=pre_output))
        except Exception as e:
            print(e)
            print('ID: ',submission[i]['ID'])
            print('pre_output: ',pre_output)
            print('gt: ',gt)
            continue

    print(len(unieval_score['Cause']),len(mover_score['Cause']),len(unieval_score['Result']),len(mover_score['Result']),len(unieval_score['Description']),len(mover_score['Description']))
    with open(total_score_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        score_data = list(reader)
    unieval_row = ['UniEval', sum(unieval_score['Cause']) / len(unieval_score['Cause']), sum(unieval_score['Result']) / len(unieval_score['Result']), sum(unieval_score['Description']) / len(unieval_score['Description'])]
    mover_row = ['MoverScore', sum(mover_score['Cause']) / len(mover_score['Cause']), sum(mover_score['Result']) / len(mover_score['Result']), sum(mover_score['Description']) / len(mover_score['Description'])]
    score_data.append(unieval_row)
    score_data.append(mover_row)
    with open(total_score_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(score_data)
    print('Classic Evaluation v2 was saved in ' +total_score_path)

if __name__ == '__main__':
    sub_files = ['/home/dh/zsc/data/test_anomalyQA/test_AQA_otter_dc.json']
    ref_path = '/home/dh/zsc/data/anomaly_dataset/test.json'
    save_path = '/home/dh/zsc/data/eval/scores'
    for sub_file in sub_files:
        print('sub_file: ', sub_file)
        model_name = sub_file.split('/')[-1].split('test_AQA_')[-1].split('.')[0]


        score_file_path = os.path.join(save_path, model_name + '_classic_score_1.csv')
        print('score_file_path: ', score_file_path)

        eval(sub_file, ref_path, score_file_path, 60)
        eval_v2(sub_file, ref_path, score_file_path)
