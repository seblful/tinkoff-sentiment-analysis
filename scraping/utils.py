import datetime


def format_date(date):

    # Create a datetime object
    datetime_object = datetime.datetime.strptime(
        date, "%Y-%m-%dT%H:%M:%S.%f%z")

    # # Format the datetime object
    # formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S.%f %z")

    return datetime_object
