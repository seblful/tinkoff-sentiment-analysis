from processors import TrainDataCreator

eng_classifier_model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


def main():
    train_data_creator = TrainDataCreator(csv_filename='pulse.csv',
                                          sample_size=5,
                                          model_name=eng_classifier_model_name)
    train_data_creator.process()
    train_data_creator.save_data_to_excel()


if __name__ == '__main__':
    main()
