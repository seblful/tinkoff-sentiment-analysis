import os
import requests
import json
import csv
import datetime


def get_api_data(url_api):
    # Get results and results content
    result = requests.get(url_api, timeout=10)
    content = result.content

    # Format bytes object to dict
    data = json.loads(content.decode('utf-8'))

    return data


def format_date(date):

    # Create a datetime object
    datetime_object = datetime.datetime.strptime(
        date, "%Y-%m-%dT%H:%M:%S.%f%z")

    # Format the datetime object
    formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S.%f %z")

    return datetime_object, formatted_datetime


def get_input_date():
    date_input = input("Enter a date in the format DD-MM-YYYY: ")
    day, month, year = [int(item) for item in date_input.split('-')]
    return datetime.date(year, month, day)


def get_lists_content(txt_file):
    with open(txt_file, 'r') as file:
        content = list(map(lambda x: x.strip('\n'), file.readlines()))

    return content


def get_csv_row_names(reaction_names, ticker_names):
    names = ['id', 'datetime', 'text', 'total_reactions', 'next_cursor']
    names.extend(reaction_names)
    names.extend(ticker_names)

    return names


def write_dict_to_csv(csv_path, item_dict, fieldnames):
    # Open the CSV file in write mode
    with open(csv_path, 'a', newline='', encoding="utf-8") as csvfile:
        # Create a csv.DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        if os.path.getsize(csv_path) == 0:
            writer.writeheader()

        # Write the rows
        writer.writerow(item_dict)
