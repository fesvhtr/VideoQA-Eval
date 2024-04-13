import json
import re

def extract_time_range(text):
    # 使用正则表达式匹配时间范围中的两个数字
    pattern = r'(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) seconds'
    match = re.search(pattern, text)
    # 检查匹配结果
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        result = [start_time, end_time]
    else:
        result = "Error"

    return result

with open('/home/dh/zsc/data/test_anomalyQA/test_funqa_timechat.json') as f:
    test_data = json.load(f)

for i_test_data in test_data:
    result = extract_time_range(i_test_data['output'])
    if result == "Error":
        print(i_test_data['visual_input'])
    # print(result)
    i_test_data['output'] = result


with open('/home/dh/zsc/data/test_anomalyQA/test_AQA_timechat_cleaned.json', 'w') as f:
    json.dump(test_data, f)

