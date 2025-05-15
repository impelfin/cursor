import json

with open('full_accom.json', encoding='utf-8') as f:
    arr = json.load(f)
    # print(arr[0])

with open('lines.json', 'w', encoding='utf-8') as f:
    for obj in arr:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')