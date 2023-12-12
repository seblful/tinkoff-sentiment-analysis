from utils import get_api_data, format_date, get_input_date, convert_data_to_json

from datetime import datetime, date, time
import pytz
import requests
import json
import os

import pandas as pd

# Set variables
# STOCKS_NAME = ''

CURRENT_CURSOR = 9999999999
RECORDINGS_LIMIT = 10
URL_API = f'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit={RECORDINGS_LIMIT}&cursor={CURRENT_CURSOR}&include=all'

# DATE_START = get_input_date()
# DATE_END = ''
DATE_START = date(2023, 12, 12)
DATETIME_START = datetime.combine(DATE_START, time.min)
DATETIME_START = pytz.utc.localize(DATETIME_START)

HOME = os.getcwd()
CSV_PATH = os.path.join(HOME, 'pulse.csv')


def process_data_json(data):
    # Retrieve next cursor and items with data
    next_cursor = data['payload']['nextCursor']

    items = data['payload']['items']

    # Iterating through each recording
    for item in items:

        item_id = item['id']

        item_date = item['inserted']
        item_datetime = format_date(item_date)

        item_reactions = item['reactions']
        item_reactions_total_counts = item_reactions['totalCount']
        item_reactions_counters = item_reactions['counters']

        for counter in item_reactions_counters:  # add counter to each reaction
            item_reactions_type = counter['type']
            item_reactions_count = counter['count']

        item_content = item['content']
        item_text = item_content['text']  # format text with tags SBER
        item_instruments = item_content['instruments']
        for instrument in item_instruments:  # take tickers
            instrument_name = instrument['ticker']
            instrument_price = instrument['price']

        break

    print(len(items))


def main():
    data = get_api_data(URL_API)
    process_data_json(data)


if __name__ == '__main__':
    main()
