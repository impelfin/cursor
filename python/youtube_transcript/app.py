import requests
import json
import os
import datetime 

def get_video_id(video_url):
    """
    YouTube 비디오 URL에서 비디오 ID를 추출합니다.
    URL 형식이 'v='를 포함하지 않을 경우를 대비하여 예외 처리를 추가했습니다.
    """
    if 'v=' in video_url:
        video_id = video_url.split("v=")[1][:11]
    else:
        print("경고: 유효한 YouTube 비디오 URL 형식이 아닙니다. 기본 ID를 사용합니다.")
        video_id = "Ks-_Mh1QhMc" 
    return video_id

def format_time(ms):
    """밀리초를 SRT 시간 포맷 (HH:MM:SS,ms)으로 변환"""
    total_seconds = int(ms / 1000)
    milliseconds = int(ms % 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

url = 'https://www.youtube.com/watch?v=JdwWgw4fq7I'
video_id = get_video_id(url)

print(f"- Youtube Video ID : {video_id}")

# Supadata API 정보
supadata_api_url = f'https://api.supadata.ai/v1/youtube/transcript?videoId={video_id}'
headers = {
    'x-api-key': 'sd_474ffea05db982b82f0bc9407f6eff9b'
}

try:
    response = requests.get(supadata_api_url, headers=headers)
    response.raise_for_status() 
    supadata_data = response.json()

    available_langs = supadata_data.get('availableLangs', [])
    transcript_content_from_api = supadata_data.get('content', [])
    primary_lang = supadata_data.get('lang', '알 수 없음')

    print("- Supadata API를 통해 가져온 정보:")
    print(f"- [기본 자막 언어] {primary_lang}")
    print(f"- [사용 가능한 언어 코드] {', '.join(available_langs)}")
    print('-' * 50)

    if not transcript_content_from_api:
        print("Supadata API 응답에 자막 내용(content)이 없습니다.")
    else:
        srt_formatted = ""
        for i, item in enumerate(transcript_content_from_api):
            start_time_ms = item.get('offset', 0)
            duration_ms = item.get('duration', 0)
            end_time_ms = start_time_ms + duration_ms
            text = item.get('text', '')

            srt_formatted += f"{i + 1}\n"
            srt_formatted += f"{format_time(start_time_ms)} --> {format_time(end_time_ms)}\n"
            srt_formatted += f"{text}\n\n"

        print("--- SRT 형식 미리보기 (처음 150자) ---")
        print(srt_formatted[:150])
        print('-' * 50)

        text_formatted = ""
        for item in transcript_content_from_api:
            text_formatted += item.get('text', '') + "\n"

        print("--- TXT 형식 미리보기 (처음 150자) ---")
        print(text_formatted[:150])
        print('-' * 50)

        # 파일 저장
        download_folder = './data'
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
            print(f"다운로드 폴더 '{download_folder}'를 생성했습니다.")

        srt_file = f'{download_folder}/{video_id}.srt'
        print('- SRT 파일 저장 : ', srt_file)
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_formatted)

        text_file = f'{download_folder}/{video_id}.txt'
        print('- TXT 파일 저장 : ', text_file)
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_formatted)

except requests.exceptions.HTTPError as e:
    print(f"HTTP 오류 발생: {e.response.status_code} - {e.response.text}")
except requests.exceptions.ConnectionError as e:
    print(f"네트워크 연결 오류 발생: {e}")
except requests.exceptions.Timeout as e:
    print(f"요청 시간 초과: {e}")
except requests.exceptions.RequestException as e:
    print(f"API 호출 중 알 수 없는 요청 오류 발생: {e}")
except json.JSONDecodeError:
    print("Supadata API 응답을 JSON으로 디코딩하는 데 실패했습니다. 응답이 JSON 형식이 아닐 수 있습니다.")
    print("API 응답 내용:", response.text[:500] if 'response' in locals() else "응답 없음")
except Exception as e:
    print(f"예상치 못한 오류 발생: {e}")
