import os
import pandas as pd
import traceback # 상세 오류 출력을 위해 추가

# 우편번호 TXT 파일들이 있는 폴더 경로
ZIPCODE_DB_FOLDER = 'zipcode_DB'
# 합쳐진 CSV 파일이 저장될 경로 및 파일명
OUTPUT_CSV_FILE = 'postcode_data_merged.csv'

def merge_zipcode_txt_to_csv():
    all_dfs = []

    txt_column_names = [
        '우편번호', '시도', '시도영문', '시군구', '시군구영문', '읍면', '읍면영문', 
        '도로명코드', '도로명', '도로명영문', '지하여부', '건물번호본번', '건물번호부번', 
        '건물관리번호', '다량배달처명', '시군구용건물명', '법정동코드', '법정동명', 
        '리명', '행정동명', '산여부', '지번본번', '읍면동일련번호', '지번부번', 
        '구우편번호', '우편번호일련번호'
    ]

    desired_columns = ['우편번호', '시도', '시군구', '읍면', '도로명', '건물번호본번', 
                       '건물번호부번', '법정동명', '리명', '지번본번', '지번부번', '시군구용건물명']

    print(f"--- '{ZIPCODE_DB_FOLDER}' 폴더에서 우편번호 TXT 파일 로드 시작 ---")

    try:
        os.makedirs(ZIPCODE_DB_FOLDER, exist_ok=True)
        processed_files_count = 0

        for filename in os.listdir(ZIPCODE_DB_FOLDER):
            if filename.endswith('.txt'):
                filepath = os.path.join(ZIPCODE_DB_FOLDER, filename)
                print(f"파일 로드 중: {filename}")
                
                try:
                    # --- 핵심 수정 부분: skiprows=1 또는 header=0 ---
                    # 파일의 첫 줄을 건너뛰고 (헤더로 간주하거나 그냥 버림), names는 직접 지정
                    # 대부분의 우편번호 TXT 파일은 헤더가 없지만, '건물번호본번' 오류는 헤더가 데이터로 읽힐 때 발생합니다.
                    # 따라서 헤더를 건너뛰는 것이 가장 안전한 접근입니다.
                    df = pd.read_csv(filepath, sep='|', encoding='utf-8', 
                                     names=txt_column_names, skiprows=1, low_memory=False) # <--- 여기를 수정했습니다.
                    
                    # 만약 `skiprows=1`로도 문제가 발생한다면, TXT 파일 내에 진짜로 숫자가 아닌 값이 섞인 것일 수 있습니다.
                    # 그럴 경우 pandas.errors.ParserError와 같은 오류가 발생할 수 있으며, 
                    # 그때는 해당 TXT 파일을 직접 열어서 확인해야 합니다.

                except UnicodeDecodeError:
                    print(f"  '{filename}' 파일이 UTF-8이 아닙니다. CP949로 재시도합니다.")
                    df = pd.read_csv(filepath, sep='|', encoding='cp949', 
                                     names=txt_column_names, skiprows=1, low_memory=False) # <--- 여기도 수정했습니다.

                except pd.errors.EmptyDataError:
                    print(f"  '{filename}' 파일이 비어있습니다. 건너뜜.")
                    continue
                except Exception as e:
                    print(f"  '{filename}' 파일 로드 중 예상치 못한 오류 발생: {e}. 건너뜜.")
                    # 특정 파일에서 오류가 발생해도 전체 프로세스를 중단하지 않고 다음 파일로 진행
                    continue

                all_dfs.append(df[desired_columns])
                processed_files_count += 1
        
        if not all_dfs:
            print(f"!!! 경고: '{ZIPCODE_DB_FOLDER}' 폴더에 처리할 TXT 파일이 없거나 모든 파일이 비어있습니다. !!!")
            return

        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"--- 모든 TXT 파일 로드 완료. 총 {len(merged_df)}개 주소 데이터. ({processed_files_count}개 파일 처리) ---")
        
        # --- 컬럼 타입 변환 및 NaN 처리 (이전 버전에서 강화된 로직 유지) ---
        # 숫자형 컬럼 처리: NaN을 0으로 채우고 정수로 변환 후 문자열로 변환
        merged_df['건물번호본번'] = merged_df['건물번호본번'].fillna(0).astype(int).astype(str)
        merged_df['건물번호부번'] = merged_df['건물번호부번'].fillna(0).astype(int).astype(str)
        merged_df['지번본번'] = merged_df['지번본번'].fillna(0).astype(int).astype(str)
        merged_df['지번부번'] = merged_df['지번부번'].fillna(0).astype(int).astype(str)
        
        # 도로명주소 및 지번주소를 구성하는 모든 문자열 컬럼을 명시적으로 문자열화 + 결측치 처리
        for col in ['시도', '시군구', '읍면', '도로명', '법정동명', '리명', '시군구용건물명', '우편번호']: # '우편번호' 추가
            if col in merged_df.columns: 
                merged_df[col] = merged_df[col].fillna('').astype(str)

        # '도로명주소_통합' 및 '지번주소_통합' 컬럼 생성
        merged_df['도로명주소_통합'] = merged_df['시도'] + ' ' + \
                                      merged_df['시군구'] + ' ' + \
                                      merged_df['도로명'] + ' ' + \
                                      merged_df['건물번호본번'] + \
                                      merged_df['건물번호부번'].apply(lambda x: f'-{x}' if x and x != '0' else '')
        
        merged_df['지번주소_통합'] = merged_df['시도'] + ' ' + \
                                    merged_df['시군구'] + ' ' + \
                                    merged_df['법정동명'] + \
                                    merged_df['리명'].apply(lambda x: f' {x}' if x else '') + ' ' + \
                                    merged_df['지번본번'] + \
                                    merged_df['지번부번'].apply(lambda x: f'-{x}' if x and x != '0' else '')
        
        # 최종 검색에 사용될 '전체주소_최종' 컬럼 생성
        merged_df['전체주소_최종'] = merged_df.apply(lambda row: 
                                                    (row['도로명주소_통합'] + ' ' + row['시군구용건물명'] if row['시군구용건물명'] else row['도로명주소_통합']), axis=1)
        
        merged_df['전체주소_최종'] = merged_df['전체주소_최종'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # 모든 데이터 타입이 적절히 변환되었는지 확인
        print("생성된 '전체주소_최종' 컬럼 샘플 (처음 5개):")
        print(merged_df['전체주소_최종'].head().tolist())

        print("\n각 컬럼의 데이터 타입 (Dtypes) 확인:")
        print(merged_df[['우편번호', '시도', '시군구', '읍면', '도로명', 
                         '건물번호본번', '건물번호부번', '법정동명', '리명', 
                         '지번본번', '지번부번', '시군구용건물명', 
                         '도로명주소_통합', '지번주소_통합', '전체주소_최종']].dtypes)

        # 결과 CSV 파일로 저장 (UTF-8 인코딩 권장)
        merged_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"--- 모든 우편번호 데이터가 '{OUTPUT_CSV_FILE}' 파일로 성공적으로 병합 및 저장되었습니다! ---")

    except Exception as e:
        print(f"!!! 오류: 우편번호 TXT 파일 병합 및 처리 중 오류 발생: {e} !!!")
        print("TXT 파일의 경로, 이름, 구분자('|'), 인코딩(encoding='utf-8' 또는 'cp949') 설정을 확인해주세요.")
        print("또한, TXT 파일의 필드 순서가 맞는지, 필요한 컬럼이 존재하는지 확인해주세요.")
        # 상세 오류 정보 출력
        traceback.print_exc()

if __name__ == "__main__":
    merge_zipcode_txt_to_csv()