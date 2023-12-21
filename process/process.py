from processors import TinkoffScraper, DataProcessor, XGBoostModel

import os

HOME = os.getcwd()
MODELS_FOLDER = os.path.join(HOME, 'models')
DATA_FOLDER = os.path.join(HOME, 'data')
LISTS_FOLDER = os.path.join(HOME, 'lists')

CSV_SAVE_PATH = os.path.join(DATA_FOLDER, 'tinkoff_data.csv')
CSV_CLEAN_SAVE_PATH = os.path.join(DATA_FOLDER, 'clean_data.csv')

TRANSFORMER_MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
TRANSFORMER_MODEL_SAVE_PATH = os.path.join(
    MODELS_FOLDER, 'rubert-tiny2-finetuned')
XGB_MODEL_SAVE_PATH = os.path.join(MODELS_FOLDER, 'xgb_dart.json')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.pkl')


def main():
    # Scraping from Tinkoff api
    # scraper = TinkoffScraper(lists_path=LISTS_FOLDER,
    #                          csv_save_path=CSV_SAVE_PATH,
    #                          date_start='21-12-2023',
    #                          date_finish='21-12-2023')
    # scraper.scrape()

    # # Process data and make initial predictions with transformer model
    # processor = DataProcessor(csv_save_path=CSV_SAVE_PATH,
    #                           csv_clean_save_path=CSV_CLEAN_SAVE_PATH,
    #                           transformer_model_name=TRANSFORMER_MODEL_NAME,
    #                           transformer_model_save_path=TRANSFORMER_MODEL_SAVE_PATH)
    # processor.process()

    # Predict with XGBoost
    xgbooster = XGBoostModel(model_save_path=XGB_MODEL_SAVE_PATH,
                             csv_clean_save_path=CSV_CLEAN_SAVE_PATH,
                             scaler_path=SCALER_PATH)

    predictions = xgbooster.predict()

    # Visualize data


if __name__ == '__main__':
    main()
