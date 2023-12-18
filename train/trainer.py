import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import Dataset, DatasetDict
import evaluate

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed, get_linear_schedule_with_warmup

from tqdm import tqdm
from accelerate import Accelerator, DistributedType


class TransformerDataset:
    def __init__(self,
                 df_path,
                 label_column,
                 text_column,
                 model_name,
                 max_length,
                 val_size,
                 random_state):

        self.val_size = val_size
        self.test_val_ratio = 0.4
        self.random_state = random_state

        self.text_column = text_column
        self.label_column = label_column
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.merged_df = self.load_df(df_path)
        self.train_df, self.val_df, self.test_df = self.create_dfs()
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_datasets()
        self.__dataset = None
        self.__tokenized_dataset = None

    def load_df(self, df_path):
        df = pd.read_excel(df_path, index_col=0).dropna()
        df = df.loc[:, ['clean_text', self.label_column]]
        df[self.label_column] = df[self.label_column].map(self.label2id)

        return df

    def create_dfs(self):
        train_df, val_test_df = train_test_split(self.merged_df,
                                                 test_size=self.val_size,
                                                 stratify=self.merged_df[self.label_column].values,
                                                 random_state=self.random_state)

        val_df, test_df = train_test_split(val_test_df,
                                           test_size=self.test_val_ratio,
                                           stratify=val_test_df[self.label_column].values,
                                           random_state=self.random_state)

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

    def tokenize_function(self, sample):
        outputs = self.tokenizer(
            sample[self.text_column],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        return outputs

    @property
    def tokenized_dataset(self):
        if self.__tokenized_dataset is None:
            tokenized_datasets = self.dataset.map(self.tokenize_function,
                                                  batched=True,
                                                  remove_columns=[
                                                      self.text_column, "__index_level_0__"]
                                                  )
            tokenized_datasets.set_format("torch")

            self.__tokenized_dataset = tokenized_datasets

        return self.__tokenized_dataset


class TransformerTrainer:
    def __init__(self,
                 tokenized_dataset,
                 model_name,
                 num_labels,
                 epochs,
                 batch_size,
                 random_seed,
                 adam_epsilon=1e-8,
                 lr=3e-5,
                 num_warmup_steps=10):

        self.epochs = epochs
        self.batch_size = batch_size

        self.tokenized_dataset = tokenized_dataset

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.create_dataloaders()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

        self.accelerator = Accelerator()

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           eps=adam_epsilon, lr=lr)

        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=len(
                                                                self.train_dataloader) * self.epochs)

        self.random_seed = random_seed

    def create_dataloaders(self):
        train_dataloader = DataLoader(
            self.tokenized_dataset["train"], shuffle=True, batch_size=self.batch_size)
        val_dataloader = DataLoader(
            self.tokenized_dataset["validation"], shuffle=False, batch_size=self.batch_size)
        test_dataloader = DataLoader(
            self.tokenized_dataset["test"], shuffle=False, batch_size=self.batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def train(self):

        set_seed(self.random_seed)
        accuracy = evaluate.load("accuracy")

        if self.accelerator.is_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        )

        progress_bar = tqdm(
            range(self.epochs * len(train_dataloader)),
            disable=not self.accelerator.is_main_process,
        )

        # Model Training
        for epoch in range(self.epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                optimizer.step()
                self.lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            model.eval()
            all_predictions = []
            all_labels = []

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)

                # gather predictions and labels from the 8 TPUs
                all_predictions.append(self.accelerator.gather(predictions))
                all_labels.append(self.accelerator.gather(batch["labels"]))

            # Concatenate all predictions and labels.
            all_predictions = torch.cat(all_predictions)[
                : len(self.tokenized_dataset["validation"])
            ]
            all_labels = torch.cat(all_labels)[: len(
                self.tokenized_dataset["validation"])]

            eval_accuracy = accuracy.compute(
                predictions=all_predictions, references=all_labels
            )

            # Use accelerator.print to print only on the main process.
            self.accelerator.print(f"epoch {epoch + 1}:", eval_accuracy)


class XGBDataset:
    pass


class XGBTrainer:
    pass
