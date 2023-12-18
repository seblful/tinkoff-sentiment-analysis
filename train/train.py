from trainer import TransformerDataset, TransformerTrainer
import os

HOME = os.getcwd()
TRAIN_DF_PATH = os.path.join(HOME, 'train_data.xlsx')

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

VALIDATION_SIZE = 0.2
MAX_LENGTH = 256

NUM_EPOCHS = 5
BATCH_SIZE = 32

RANDOM_STATE = 11
RANDOM_SEED = 12


def main():
    transf_dataset = TransformerDataset(df_path=TRAIN_DF_PATH,
                                        label_column='labels',
                                        text_column='clean_text',
                                        model_name=MODEL_NAME,
                                        max_length=MAX_LENGTH,
                                        val_size=VALIDATION_SIZE,
                                        random_state=RANDOM_STATE)
    # print(len(transf_dataset.train_df), len(
    #     transf_dataset.val_df), len(transf_dataset.test_df))

    trainer = TransformerTrainer(tokenized_dataset=transf_dataset.tokenized_dataset,
                                 model_name=MODEL_NAME,
                                 num_labels=3,
                                 epochs=NUM_EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 random_seed=RANDOM_SEED)

    trainer.train()


if __name__ == '__main__':
    main()
