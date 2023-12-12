import requests
import json
import csv
import datetime


def get_api_data(url_api):
    # Get results and results content
    result = requests.get(url_api)
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


def write_to_csv(file_path, *args):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(args)


def write_to_csv(file_path, *args, **kwargs):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in args:
            writer.writerow(item)
        for key, value in kwargs.items():
            writer.writerow({key: value})
