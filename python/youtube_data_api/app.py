import os
import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# YouTube Data API 사용을 위한 OAuth 2.0 스코프
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

def get_authenticated_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            
            # --- 변경된 부분 시작 ---
            # 웹 브라우저를 자동으로 열지 않도록 open_browser=False를 추가하고,
            # 포트를 0으로 설정하여 사용 가능한 포트를 자동으로 찾게 합니다.
            auth_url, _ = flow.authorization_url(prompt='consent', include_granted_scopes='true')
            print("--- Google 인증을 진행해주세요 ---")
            print(f"아래 URL을 웹 브라우저에 복사하여 붙여넣으세요:\n{auth_url}")
            print("\n인증 완료 후, 브라우저에 표시되는 '인증 코드'를 여기에 입력해주세요:")
            
            # 사용자가 수동으로 인증 코드를 입력하게 합니다.
            code = input("인증 코드: ").strip()
            flow.fetch_token(code=code) # 입력받은 코드로 토큰을 가져옵니다.
            creds = flow.credentials
            # --- 변경된 부분 끝 ---

        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('youtube', 'v3', credentials=creds)

# (나머지 get_video_id 및 download_captions 함수와 main 부분은 동일합니다.)

def get_video_id(video_url):
    """YouTube URL에서 비디오 ID를 추출합니다."""
    # YouTube URL 형식이 다양할 수 있으므로, 좀 더 robust하게 처리합니다.
    if "v=" in video_url:
        video_id = video_url.split("v=")[1]
        # 추가적인 파라미터가 있을 수 있으므로, &를 기준으로 잘라냅니다.
        if "&" in video_id:
            video_id = video_id.split("&")[0]
        return video_id[:11]  # 비디오 ID는 보통 11자입니다.
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1]
        if "?" in video_id:
            video_id = video_id.split("?")[0]
        return video_id
    else:
        return None # 유효하지 않은 URL

def download_captions(youtube, video_id, download_folder='./data'):
    """YouTube Data API를 사용하여 자막을 다운로드합니다."""
    try:
        # 1. 자막 트랙 목록 가져오기
        captions_list_request = youtube.captions().list(
            part='snippet',
            videoId=video_id
        )
        captions_list_response = captions_list_request.execute()

        print(f"'{video_id}' 비디오의 사용 가능한 자막 트랙:")
        for item in captions_list_response.get('items', []):
            print(f"- ID: {item['id']}, 언어: {item['snippet']['language']}, 이름: {item['snippet'].get('name', 'N/A')}")

        # 2. 한국어 자막 트랙 ID 찾기 (또는 다른 언어)
        target_caption_id = None
        for item in captions_list_response.get('items', []):
            if item['snippet']['language'] == 'ko':
                target_caption_id = item['id']
                print(f"한국어 자막 ID를 찾았습니다: {target_caption_id}")
                break
        if not target_caption_id:
             # 한국어 자막이 없으면, 생성된 자막을 찾습니다.
            for item in captions_list_response.get('items', []):
                if item['snippet']['language'] == 'a.ko':  # a.ko는 자동 생성된 한국어 자막을 의미합니다.
                    target_caption_id = item['id']
                    print(f"자동 생성된 한국어 자막 ID를 찾았습니다: {target_caption_id}")
                    break
        if not target_caption_id:
            print("한국어 자막 트랙을 찾을 수 없습니다. 다운로드할 수 있는 자막이 없습니다.")
            return

        # 3. 자막 다운로드
        caption_download_request = youtube.captions().download(
            id=target_caption_id,
            tfmt='srt'  # SRT 형식으로 다운로드 (다른 형식도 지원: vtt, ttml)
        )
        caption_content = caption_download_request.execute()

        # 4. 다운로드 폴더 생성 (필요한 경우)
        os.makedirs(download_folder, exist_ok=True)

        # 5. 파일로 저장
        srt_file = os.path.join(download_folder, f"{video_id}.ko.srt")
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(caption_content)
        print(f"자막이 '{srt_file}'으로 성공적으로 다운로드되었습니다.")


    except Exception as e:
        print(f"자막 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    # 1. 인증된 YouTube Data API 서비스 객체 생성
    youtube = get_authenticated_service()

    # 2. 비디오 URL 및 ID
    video_url = 'https://www.youtube.com/watch?v=Ks-_Mh1QhMc'  # 테스트 URL
    video_id = get_video_id(video_url)

    if not video_id:
        print("유효하지 않은 YouTube URL입니다.")
    else:
        # 3. 자막 다운로드
        download_captions(youtube, video_id)