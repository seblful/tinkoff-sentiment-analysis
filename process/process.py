from processors import TinkoffScraper

import os

HOME = os.getcwd()
LISTS_PATH = os.path.join(HOME, 'lists')
CSV_SAVE_PATH = os.path.join(HOME, 'tinkoff_data.csv')


def main():
    # Scraping from Tinkoff api
    scraper = TinkoffScraper(lists_path=LISTS_PATH,
                             csv_save_path=CSV_SAVE_PATH,
                             date_start='16-12-2023',
                             date_finish='18-12-2023')
    scraper.scrape()


if __name__ == '__main__':
    main()
6
