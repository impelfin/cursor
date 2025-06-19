from googleapiclient.discovery import build

API_KEY = 'AIzaSyC19Q_v0lO1mHs_dJYx-_wDCsNUDJignboY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def youtube_search(query, max_results=5):
    """
    Performs a YouTube search and returns a list of video titles and IDs.
    """
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',  # Search only for videos
        maxResults=max_results
    )

    response = request.execute()

    videos = []
    for item in response.get('items', []):
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        videos.append({'id': video_id, 'title': title})
    
    return videos

if __name__ == '__main__':
    search_query = 'Python programming tutorial'
    results = youtube_search(search_query, max_results=10)

    if results:
        print(f"Search results for '{search_query}':")
        for video in results:
            print(f"  Title: {video['title']}")
            print(f"  Video ID: {video['id']}\n")
    else:
        print("No results found.")
