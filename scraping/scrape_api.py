import requests
import json

# Set variables
current_cursor = 9999999999
limit = 10
url = f'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit={limit}&cursor={current_cursor}&include=all'

# Get results and results content
result = requests.get(url)
content = result.content

# Format bytes object to dict
data = json.loads(content.decode('utf-8'))

# Retrieve next cursor and items with data
next_cursor = data['payload']['nextCursor']

items = data['payload']['items']

# Iterating through each recording
for item in items:

    item_id = item['id']

    item_data = item['inserted']

    item_reactions = item['reactions']
    item_reactions_total_counts = item_reactions['totalCount']
    item_reactions_counters = item_reactions['counters']
    for counter in item_reactions_counters:  # add counter to each reaction
        pass

    item_content = item['content']
    text = item_content['text']  # format text with tags SBER
    instruments = item_content['instruments']
    for instrument in instruments:  # take tickers
        pass

print(len(items))
