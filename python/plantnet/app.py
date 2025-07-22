import requests
import json
from pprint import pprint
import os 

API_KEY = os.getenv("PLANTNET_API_KEY") 
if not API_KEY:
    raise ValueError("PLANTNET_API_KEY 환경 변수가 설정되지 않았습니다.")

PROJECT = "all"; # You can choose a more specific flora, see: /docs/newfloras
api_endpoint = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"

image_path_1 = "./data/asd.png"
image_data_1 = open(image_path_1, 'rb')

# image_path_2 = "./data/pinetree.png"
# image_data_2 = open(image_path_2, 'rb')

data = { 'organs': ['flower'] }
# data = { 'organs': ['flower', 'leaf'] }

files = [
    ('images', (image_path_1, image_data_1)),
#     ('images', (image_path_2, image_data_2))
]

req = requests.Request('POST', url=api_endpoint, files=files, data=data)
prepared = req.prepare()

s = requests.Session()
response = s.send(prepared)
json_result = json.loads(response.text)

pprint(response.status_code)
pprint(json_result)
