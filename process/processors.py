import os
import csv
import json
import requests

from numpy import random

from time import sleep
from datetime import datetime, date, time
import pytz


class TinkoffScraper:
    def __init__(self,
                 lists_path,
                 csv_save_path,
                 date_start=None,
                 date_finish=None,
                 recordings_limit=100,
                 start_cursor=9_999_999_999):

        # Instantiate paths
        self.reactions_names_path = os.path.join(
            lists_path, 'reactions_names.txt')
        self.ticker_names_path = os.path.join(
            lists_path, 'pop_ticker_names.txt')
        self.csv_save_path = csv_save_path

        # Extract ticker and reaction names
        self.reaction_names = self.get_lists_content(self.reactions_names_path)
        self.ticker_names = self.get_lists_content(self.ticker_names_path)
        self.csw_row_names = self.get_csv_row_names()

        # Instantiate time
        self.datetime_now = pytz.utc.localize(datetime.now())
        self.date_start = self.split_date(
            date_start) if date_start is not None else self.get_input_date()
        self.datetime_start = pytz.utc.localize(
            datetime.combine(self.date_start, time.min))

        self.date_finish = self.split_date(
            date_finish) if date_finish is not None else self.get_input_date()
        self.datetime_finish = pytz.utc.localize(
            datetime.combine(self.date_finish, time.max))

        # Url api to scrape
        self.recordings_limit = recordings_limit
        self.start_cursor = start_cursor
        self.url_api = f'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit={recordings_limit}&cursor={start_cursor}&include=all'

        self.stop_scraping = False

    def get_lists_content(self, txt_file_path):
        with open(txt_file_path, 'r') as file:
            content = list(map(lambda x: x.strip('\n'), file.readlines()))

        return content

    def get_csv_row_names(self):
        names = ['id', 'datetime', 'text', 'total_reactions', 'next_cursor']
        names.extend(self.reaction_names)
        names.extend(self.ticker_names)

        return names

    def split_date(self, input_date):
        day, month, year = [int(item) for item in input_date.split('-')]
        return date(year, month, day)

    def get_input_date(self):
        input_date = input("Enter a date in the format 'DD-MM-YYYY': ")
        return self.split_date(input_date)

    def get_api_data(self):
        # Get results and results content
        result = requests.get(self.url_api, timeout=10)
        content = result.content

        # Format bytes object to dict
        data = json.loads(content.decode('utf-8'))

        return data

    def format_date(self, date):

        # Create a datetime object
        datetime_object = datetime.strptime(
            date, "%Y-%m-%dT%H:%M:%S.%f%z")

        # Format the datetime object
        formatted_datetime = datetime_object.strftime(
            "%Y-%m-%d %H:%M:%S.%f %z")

        return datetime_object, formatted_datetime

    def write_dict_to_csv(self, item_dict):
        # Open the CSV file in write mode
        with open(self.csv_save_path, 'a', newline='', encoding="utf-8") as csvfile:
            # Create a csv.DictWriter object
            writer = csv.DictWriter(csvfile, fieldnames=self.csw_row_names)

            # Write the header
            if os.path.getsize(self.csv_save_path) == 0:
                writer.writeheader()

            # Write the rows
            writer.writerow(item_dict)

    def process_and_write_data(self, data):
        # Retrieve next cursor and items with data
        next_cursor = data['payload']['nextCursor']
        # Items with current cursor
        items = data['payload']['items']

        # Iterating through each recording
        for item in items:

            # Dicts for storing reactions and tickers
            reaction_names_dict = {key: 0 for key in self.reaction_names}
            ticker_names_dict = {key: 0 for key in self.ticker_names}

            item_id = item['id']

            item_date = item['inserted']
            item_datetime, item_datetime_str = self.format_date(item_date)

            if item_datetime < self.datetime_start:
                self.stop_scraping = True
                break

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
                         'total_reactions': item_reactions_total_counts,
                         'next_cursor': next_cursor}

            merged_item_dict = dict(
                item_dict, **reaction_names_dict, **ticker_names_dict)

            # Write data to csv
            if item_datetime < self.datetime_finish:
                self.write_dict_to_csv(merged_item_dict)

        return next_cursor, item_datetime

    def scrape(self):
        # Get data from start to finish
        sum_days_to_scrape = (self.datetime_finish -
                              self.datetime_start).days
        print(f"It is {sum_days_to_scrape} days to scrape.")

        while self.stop_scraping is False:
            # Get chank of json data
            data = self.get_api_data()
            # Process data and write to csv
            next_cursor, item_datetime = self.process_and_write_data(data)

            # Create new url_api with new cursor
            self.url_api = f'https://www.tinkoff.ru/api/invest-gw/social/post/feed/v1/post/instrument/SBER?sessionId=QxZLiUIV31WxIZ4WonMwyIGI3UqG0zFO.ds-prod-api-101&appName=socialweb&appVersion=1.380.0&origin=web&platform=web&limit={self.recordings_limit}&cursor={next_cursor}&include=all'

            # Count how many days
            days_gone = (self.datetime_finish - item_datetime).days if item_datetime < self.datetime_finish else 0

            print(f"It was scraped {days_gone}/{sum_days_to_scrape} days.")

            # Sleep some time to do not be blocked
            sleep(random.uniform(2, 4))


class TransformerModel:
    pass


class XGBoostModel:
    pass


class Visualizer:
    pass
