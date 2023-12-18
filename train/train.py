from trainer import TransformerDataset
import os

HOME = os.getcwd()
TRAIN_DF_PATH = os.path.join(HOME, 'train_data.xlsx')
VALIDATION_SIZE = 0.2
MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
RANDOM_STATE = 11


def main():
    d = TransformerDataset(df_path=TRAIN_DF_PATH,
                           val_size=VALIDATION_SIZE,
                           model_name=MODEL_NAME,
                           random_state=RANDOM_STATE)

    print(len(d.train_dataset), len(d.val_dataset), len(d.test_dataset))


if __name__ == '__main__':
    main()
