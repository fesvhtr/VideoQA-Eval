import json


with open('/home/dh/zsc/data/anomaly_dataset/test.json') as f:
    gt_data = json.load(f)
with open('/home/dh/zsc/data/test_anomalyQA/test_AQA_timechat.json') as f:
    test_data = json.load(f)
print(len(gt_data), len(test_data))

import re


def convert_time_to_seconds(time_str):
    # 提取分钟和秒钟的部分
    minutes = int(time_str[0:2])
    seconds = int(time_str[2:4])

    # 计算总秒数
    total_seconds = minutes * 60 + seconds

    return total_seconds

def parse_time_intervals(string):
    # 使用正则表达式提取时间段
    pattern = r'\[(\d+),(\d+)\]'
    matches = re.findall(pattern, string)

    seconds_list = []
    # 转换为秒数
    for match in matches:
        # print(match)
        s1 = convert_time_to_seconds(match[0])
        s2 = convert_time_to_seconds(match[1])
        seconds_list.append([s1,s2])

    return seconds_list


def calculate_iou(interval1, interval2):
    # 计算交集的起始和结束时间
    intersection_start = max(interval1[0], interval2[0])
    intersection_end = min(interval1[1], interval2[1])

    # 计算交集和并集的长度
    intersection_length = max(0, intersection_end - intersection_start)
    union_length = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])

    # 计算交并比
    iou = intersection_length / union_length if union_length > 0 else 0

    return iou

acc = 0
cnt = 0
for i in range(len(gt_data)):
    if gt_data[i]['task'] == 'Timestamp':

        input_string = gt_data[i]['output']
        parsed_intervals = parse_time_intervals(input_string)
        answer_interval = test_data[i]['output']
        iou_scores = [calculate_iou(answer_interval, interval) for interval in parsed_intervals]
        sum_iou = sum(iou_scores)
        acc += sum_iou
        cnt +=1

print(acc, cnt, acc/cnt)

