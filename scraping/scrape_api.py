from utils import get_api_data, format_date, get_input_date, get_lists_content, get_csv_row_names, write_dict_to_csv

import os
from datetime import datetime, date, time
import pytz


# Set variables
# STOCK_NAME = ''

# Instantiate paths
HOME = os.getcwd()
LISTS_FOLDER = os.path.join(HOME, 'lists')
TICKER_NAMES_PATH = os.path.join(LISTS_FOLDER, 'pop_ticker_names.txt')
REACTIONS_NAMES_PATH = os.path.join(LISTS_FOLDER, 'reactions_names.txt')
CSV_PATH = os.path.join(HOME, 'pulse.csv')

# Extract ticker and reaction names
REACTION_NAMES = get_lists_content(REACTIONS_NAMES_PATH)
POP_TICKER_NAMES = get_lists_content(TICKER_NAMES_PATH)
CSV_ROW_NAMES = get_csv_row_names(REACTION_NAMES, POP_TICKER_NAMES)

# DATE_START = get_input_date()
# DATE_END = ''
DATE_START = date(2023, 12, 12)
DATETIME_START = datetime.combine(DATE_START, time.min)
DATETIME_START = pytz.utc.localize(DATETIME_START)


def process_and_write_data(data):
    # Retrieve next cursor and items with data
    next_cursor = data['payload']['nextCursor']
    # Items with current cursor
    items = data['payload']['items']

    # Iterating through each recording
    for item in items:

        # Dicts for storing reactions and tickers
        reaction_names_dict = {key: 0 for key in REACTION_NAMES}
        ticker_names_dict = {key: 0 for key in POP_TICKER_NAMES}

        item_id = item['id']

        item_date = item['inserted']
        item_datetime, item_datetime_str = format_date(item_date)

        item_reactions = item['reactions']
        item_reactions_total_counts = item_reactions['totalCount']
        item_reactions_counts = item_reactions['counters']

        for counter in item_reactions_counts:  # add counter to each reaction
            item_reactions_type = counter['type']
            item_reactions_count = counter['count']

            if item_reactions_type in reaction_names_dict.keys():
                reaction_names_dict[item_reactions_type] = item_reactions_count

        item_content = item['content']
        item_text = item_content['text']  # format text with tags SBER
        item_instruments = item_content['instruments']
        for instrument in item_instruments:  # take tickers
            instrument_name = instrument['ticker']
            instrument_price = instrument['price']

            if instrument_name in ticker_names_dict.keys():
                ticker_names_dict[instrument_name] = instrument_price

        item_dict = {'id': item_id,
                     'datetime': item_datetime_str,
                     'text': item_text,
                     'total_reactions': item_reactions_total_counts}

        merged_item_dict = dict(
            item_dict, **reaction_names_dict, **ticker_names_dict)

        # print(merged_item_dict, type(merged_item_dict))

        # Write data to csv
        write_dict_to_csv(csv_path=CSV_PATH,
                          item_dict=merged_item_dict,
                          fieldnames=CSV_ROW_NAMES)

    return next_cursor, item_datetime, merged_item_dict

# item_id, item_datetime, item_text, item_reactions_total_counts, *item_reactions_type, *instrument_name
# item_id, item_datetime_str, item_text, item_reactions_total_counts, *item_reactions_count, *instrument_price


def main():
    # Instantiate variables and create url api
    recordings_limit = 100
    start_cursor = 9999999999
    url_api = f'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit={recordings_limit}&cursor={start_cursor}&include=all'

    # Get data from now to DATETIME_START
    item_datetime = DATETIME_START
    while DATETIME_START <= item_datetime:
        # Get chank of json data
        data = get_api_data(url_api)
        # Process data and write to csv
        next_cursor, item_datetime, merged_item_dict = process_and_write_data(
            data)

        # Create new url_api with new cursor
        url_api = f'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit={recordings_limit}&cursor={next_cursor}&include=all'


if __name__ == '__main__':
    main()
