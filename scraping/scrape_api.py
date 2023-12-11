import requests
import json

actual_url = 'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit=100&include=all'
next_url = 'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit=50&cursor=9703535&include=all'

result = requests.get(actual_url)
content = result.content

data = json.loads(content.decode('utf-8'))

items = data['payload']['items']

next_cursor = data['payload']['nextCursor']
