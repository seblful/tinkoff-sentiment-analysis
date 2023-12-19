from trainer import TransformerDataset, TransformerTrainer, XGBostProcessor
import os

HOME = os.getcwd()
TRAIN_DF_PATH = os.path.join(HOME, 'train_data.xlsx')

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
FINETUNED_MODEL_PATH = os.path.join(HOME, 'rubert-tiny2-finetuned')

VALIDATION_SIZE = 0.2
MAX_LENGTH = 24

NUM_EPOCHS = 1
BATCH_SIZE = 32

RANDOM_STATE = 11
RANDOM_SEED = 12


def main():
    # Create TransformerDataset
    print("Creating and preparing dataset...")
    transf_dataset = TransformerDataset(df_path=TRAIN_DF_PATH,
                                        label_column='labels',
                                        text_column='clean_text',
                                        model_name=MODEL_NAME,
                                        max_length=MAX_LENGTH,
                                        val_size=VALIDATION_SIZE,
                                        random_state=RANDOM_STATE)
    print("Dataset has been created.")

    # Get dfs from transf_dataset for XGBostProcessor
    train_df, val_df, test_df = transf_dataset.train_df, transf_dataset.val_df, transf_dataset.test_df

    # Train with TransformerTrainer
    print("Training Transformer model...")
    transf_trainer = TransformerTrainer(tokenized_dataset=transf_dataset.tokenized_dataset,
                                        model_name=MODEL_NAME,
                                        num_labels=3,
                                        epochs=NUM_EPOCHS,
                                        batch_size=BATCH_SIZE,
                                        random_seed=RANDOM_SEED,
                                        save_model_path=FINETUNED_MODEL_PATH)
    transf_trainer.train()
    print("Transformer model has been trained.")

    # Get transf_model for XGBostProcessor
    transf_model = transf_trainer.model
    transf_tokenizer = transf_dataset.tokenizer

    print(f"Test accuracy: {transf_trainer.test()}.")
    transf_trainer.save_model()
    print("Transformer model has been saved.")

    # Train XGBostProcessor
    print("Training XGBoost model...")
    xgb = XGBostProcessor(train_df=train_df,
                          val_df=val_df,
                          test_df=test_df,
                          transf_model=transf_model,
                          transf_tokenizer=transf_tokenizer)


if __name__ == '__main__':
    main()
