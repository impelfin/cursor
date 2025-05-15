import csv
import numpy as np
import random
import math

# 파일 경로
csv_path = 'data.csv'
outfile_path = 'data_filled.csv'

# 데이터 읽기
data = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        year = int(row[0])
        value = row[1].strip()
        data.append([year, value if value else None])

# 2025년의 값이 500000이 되도록 전체 곡선을 맞추기 위한 보간/증가 함수

def get_first_value(data):
    for year, value in data:
        if value is not None and value != '':
            return int(year), float(value)
    return None, None

first_year, first_value = get_first_value(data)
mid_year, mid_value = 2025, 493408
last_year, last_value = 2100, 1240800
max_year = last_year
all_years = [row[0] for row in data]
data_dict = {year: value for year, value in data}

filled_data = []

# 2005~2009년 실제 값 추출
known_years = [row[0] for row in data if row[1] not in (None, '') and row[0] <= 2009]
known_values = [float(row[1]) for row in data if row[1] not in (None, '') and row[0] <= 2009]

for year in range(min(all_years), max_year+1):
    # 이미 값이 있으면 그대로
    if year in data_dict and data_dict[year] not in (None, ''):
        filled_data.append([year, int(data_dict[year])])
        continue
    # 값이 없는 연도만 보간/생성
    if year <= 2009:
        # 2006~2009년은 실제 값 사용(위 if문에서 이미 처리됨, 안전하게 남겨둠)
        idx = known_years.index(year)
        value = int(known_values[idx])
        filled_data.append([year, value])
    elif year > 2009 and year <= mid_year:
        # 2010~2025: 2009년 값에서 2025년 500000까지, 초반 급격히 증가 후 완만 (power curve)
        start_year = 2009
        start_value = float([row[1] for row in data if row[0]==start_year][0])
        t = (year - start_year) / (mid_year - start_year)
        a = 0.4  # 0.3~0.5 사이면 초반 가파름
        interp = start_value + (mid_value - start_value) * pow(t, a)
        noise = random.uniform(-0.01, 0.01) * (mid_value - start_value) * (1-t)
        value = interp + noise
        if year == mid_year:
            value = mid_value
        value = int(round(value))
        if filled_data:
            value = max(value, int(filled_data[-1][1])+1)
        filled_data.append([year, value])
    else:
        # 2026~2100: 2025년 값에서 2100년 1200000까지, 완만한 곡선 (power curve)
        t = (year - mid_year) / (last_year - mid_year)
        b = 0.8  # 0.7~0.9 사이면 후반 완만
        interp = mid_value + (last_value - mid_value) * pow(t, b)
        noise = random.uniform(-0.01, 0.01) * (last_value - mid_value) * (1-t)
        value = interp + noise
        if year == last_year:
            value = last_value
        value = int(round(value))
        if filled_data:
            value = max(value, int(filled_data[-1][1])+1)
        filled_data.append([year, value])

# 결과 저장
def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

write_csv(outfile_path, filled_data)

print("finished")
