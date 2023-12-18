import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_metric, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# def model_config():
#     cfg_dictionary = {
#         "data_path": "../input/financial-sentiment-analysis/data.csv",
#         "model_path": "/kaggle/working/bert_model.h5",
#         "model_type": "transformer",

#         "test_size": 0.1,
#         "validation_size": 0.2,
#         "train_batch_size": 32,
#         "eval_batch_size": 32,

#         "epochs": 5,
#         "adam_epsilon": 1e-8,
#         "lr": 3e-5,
#         "num_warmup_steps": 10,

#         "max_length": 128,
#         "random_seed": 42,
#         "num_labels": 3,
#         "model_checkpoint": "roberta-base",
#     }
#     cfg = ml_collections.FrozenConfigDict(cfg_dictionary)

#     return cfg


# cfg = model_config()

# tokenizer = AutoTokenizer.from_pretrained("seara/rubert-tiny2-russian-sentiment")
# model = AutoModelForSequenceClassification.from_pretrained("seara/rubert-tiny2-russian-sentiment")


class TransformerDataset:
    def __init__(self,
                 df_path,
                 val_size,
                 model_name,
                 random_state):

        self.val_size = 0.2
        self.test_val_ratio = 0.4

        self.random_state = random_state
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.model_name = model_name

        self.merged_df = self.load_df(df_path)

        self.train_df, self.val_df, self.test_df = self.create_dfs()

        self.train_dataset, self.val_dataset, self.test_dataset = self.create_datasets()

        self.__dataset = None

    def load_df(self, df_path):
        df = pd.read_excel(df_path, index_col=0)
        df = df.loc[:, ['clean_text', 'label']]
        df['label'] = df['label'].map(self.label2id)

        return df

    def create_dfs(self):
        train_df, val_test_df = train_test_split(
            self.merged_df, test_size=self.val_size, stratify=self.merged_df.label.values)
        val_df, test_df = train_test_split(
            val_test_df, test_size=self.test_val_ratio, stratify=val_test_df.label.values)

        return train_df, val_df, test_df

    def create_datasets(self):
        train_dataset = Dataset.from_pandas(self.train_df)
        val_dataset = Dataset.from_pandas(self.val_df)
        test_dataset = Dataset.from_pandas(self.test_df)

        return train_dataset, val_dataset, test_dataset

    @property
    def dataset(self):
        if self.__dataset is None:
            dataset_dict = {
                'train': self.train_dataset,
                'validation': self.val_dataset,
                'test': self.test_dataset
            }

            self.__dataset = DatasetDict(dataset_dict)

        return self.__dataset


class TransformerTrainer:
    pass


class XGBDataset:
    pass


class XGBTrainer:
    pass
