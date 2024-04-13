import json
import os
import re
import csv

def match_score(text):
    pattern = r"Score: (\d+)/\d+"
    # Use re.search to find the pattern in the text
    match = re.search(pattern, text)
    # Extract the score if a match is found
    if match:
        score = match.group(1)
        return int(score)
    else:
        print("Score not found in the text.")
        return -1

def cal_mmEval(file,output_file):
    model_name = file.split('mmaEval_output_')[-1].split('.')[0]
    with open(file) as f:
        mmEval_data = json.load(f)
    with open(output_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        score_data = list(reader)
    scores = {}
    scores['Description'], scores['Result'], scores['Cause'] = [], [], []
    for i in mmEval_data:
        if i['task'] == 'Description':
            add_score = match_score(i['output'])
            if add_score == -1:
                continue
            scores['Description'].append(add_score)
        elif i['task'] == 'Result':
            add_score = match_score(i['output'])
            if add_score == -1:
                continue
            scores['Result'].append(add_score)
        elif i['task'] == 'Cause':
            add_score = match_score(i['output'])
            if add_score == -1:
                continue
            scores['Cause'].append(add_score)
    score_row = [model_name, sum(scores['Description'])/len(scores['Description']), sum(scores['Result'])/len(scores['Result']), sum(scores['Cause'])/len(scores['Cause'])]
    score_data.append(score_row)
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(score_data)
    print('model: ' + model_name + 'score was saved')

if __name__ == '__main__':
    mmEval_files = ['/home/dh/zsc/data/eval/mmaEval/mmaEval_output_CUVA.json']
    output_file = '/home/dh/zsc/data/eval/mmaEval/mmEval_score.csv'
    for mmEval_file in mmEval_files:
        cal_mmEval(mmEval_file,output_file)
    print('mmEval was saved in ' + output_file)
