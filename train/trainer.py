import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import datasets
from datasets import Dataset, DatasetDict
import evaluate

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed, get_linear_schedule_with_warmup

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
        self.target_columns = ['clean_text', self.label_column]
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.merged_df = self.load_df(df_path)
        self.train_df, self.val_df, self.test_df = self.create_dfs()
        self.tiny_train_df, self.tiny_val_df, self.tiny_test_df = self.create_tiny_dfs()
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_datasets()
        self.__dataset = None
        self.__tokenized_dataset = None

    def load_df(self, df_path):
        df = pd.read_excel(df_path, index_col=0).dropna()
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

    def create_tiny_dfs(self):
        train_df, val_df, test_df = [df.loc[:, self.target_columns]
                                     for df in [self.train_df, self.val_df, self.test_df]]

        return train_df, val_df, test_df

    def create_datasets(self):

        train_dataset = Dataset.from_pandas(self.tiny_train_df)
        val_dataset = Dataset.from_pandas(self.tiny_val_df)
        test_dataset = Dataset.from_pandas(self.tiny_test_df)

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
                 save_model_path,
                 adam_epsilon=1e-8,
                 lr=3e-5,
                 num_warmup_steps=10):

        self.epochs = epochs
        self.batch_size = batch_size

        self.save_model_path = save_model_path

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

        self.accuracy = evaluate.load("accuracy")

        self.progress_bar = tqdm(range(self.epochs * len(self.train_dataloader)),
                                 disable=not self.accelerator.is_main_process)

        self.random_seed = random_seed

    def create_dataloaders(self):
        train_dataloader = DataLoader(
            self.tokenized_dataset["train"], shuffle=True, batch_size=self.batch_size)
        val_dataloader = DataLoader(
            self.tokenized_dataset["validation"], shuffle=False, batch_size=self.batch_size)
        test_dataloader = DataLoader(
            self.tokenized_dataset["test"], shuffle=False, batch_size=self.batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def set_logging(self):
        if self.accelerator.is_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    def prepare_mod(self):
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(self.model,
                                                                                                                                self.optimizer,
                                                                                                                                self.train_dataloader,
                                                                                                                                self.val_dataloader,
                                                                                                                                self.test_dataloader)

    def prepare_all(self):
        # Preparation for training
        set_seed(self.random_seed)
        self.set_logging()
        self.prepare_mod()

    def train_step(self):
        self.model.train()

        for step, batch in enumerate(self.train_dataloader):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress_bar.update(1)

    def val_step(self):
        self.model.eval()

        all_predictions = []
        all_labels = []

        for step, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            # gather predictions and labels from the 8 TPUs
            all_predictions.append(self.accelerator.gather(predictions))
            all_labels.append(self.accelerator.gather(batch["labels"]))

        # Concatenate all predictions and labels.
        all_predictions = torch.cat(all_predictions)[
            :len(self.tokenized_dataset["validation"])]
        all_labels = torch.cat(all_labels)[: len(
            self.tokenized_dataset["validation"])]

        val_accuracy = self.accuracy.compute(predictions=all_predictions,
                                             references=all_labels)

        return val_accuracy

    def train(self):
        self.prepare_all()

        # Model Training
        for epoch in range(self.epochs):
            self.train_step()
            val_accuracy = self.val_step()

            # Use accelerator.print to print only on the main process.
            self.accelerator.print(f"Epoch {epoch + 1}:", val_accuracy)

    def test(self):
        self.model.eval()

        all_predictions = []
        all_labels = []

        for step, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            # gather predictions and labels from the 8 TPUs
            all_predictions.append(self.accelerator.gather(predictions))
            all_labels.append(self.accelerator.gather(batch["labels"]))

        # Concatenate all predictions and labels.
        all_predictions = torch.cat(all_predictions)[
            :len(self.tokenized_dataset["test"])]
        all_labels = torch.cat(all_labels)[: len(
            self.tokenized_dataset["test"])]

        test_accuracy = self.accuracy.compute(predictions=all_predictions,
                                              references=all_labels)

        return test_accuracy

    def save_model(self):
        self.model.save_pretrained(self.save_model_path)


class XGBostProcessor:
    def __init__(self,
                 train_df,
                 val_df,
                 test_df,
                 transf_model,
                 transf_tokenizer):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tfidf_vectorizer = TfidfVectorizer()
        self.transf_model = transf_model
        self.transf_tokenizer = transf_tokenizer

        self.target_columns = ['datetime', 'total_reactions', 'like',
                               'rocket', 'buy-up', 'dislike', 'not-convinced', 'get-rid', 'SBER',
                               'SBERP', 'GAZP', 'LKOH', 'VTBR', 'MOEX', 'ROSN', 'YNDX', 'TCSG', 'NVTK',
                               'USDRUB', 'TATN', 'GMKN', 'MGNT', 'POLY', 'VKCO', 'CHMF', 'MTSS',
                               'OZON', 'POSI', 'clean_text']
        self.label_column = 'labels'

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.formatted_train_df = self.preprocess_df(
            df=self.train_df, name_of_set='train')
        self.formatted_val_df = self.preprocess_df(
            df=val_df, name_of_set='validation')
        self.formatted_test_df = self.preprocess_df(
            df=test_df, name_of_set='test')

        # # Create and fit StandardScaler
        # self.standart_scaler = StandardScaler().set_output(transform="pandas")
        # self.standart_scaler.fit(self.formatted_train_df)

    def preprocess_df(self, df, name_of_set):
        # Leave only target columns
        df = df.loc[:, self.target_columns]

        # Format datetime column to datetime format
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Create predictions with transf_model
        tqdm.pandas(desc=f'Classifying text of {name_of_set} set for XGBoost')
        df[['predicted_label', 'predicted_score']] = df.loc[:,
                                                            'clean_text'].progress_apply(self.classify_text)
        # Preprocess text (vectorization) and rename column
        df['clean_text'] = self.tfidf_vectorizer.fit_transform(
            df['clean_text']).toarray()
        df.rename(columns={"clean_text": "clean_text_tfidf"}, inplace=True)

        # # Scale columns with StandardScaler()
        # print(self.standart_scaler.transform(df))

        return df

    def classify_text(self, text):
        # Decreaze length of text to model maximum
        text = text[:500]

        encoded_input = self.transf_tokenizer(
            text, return_tensors='pt').to(self.device)

        # Get the logits
        output = self.transf_model(**encoded_input)
        logits = output.logits
        probabilities = F.softmax(logits, dim=1)

        # Access the id2label mapping
        predicted_class_id = logits.argmax().item()
        predicted_class = self.transf_model.config.id2label[predicted_class_id]
        predicted_probability = probabilities.squeeze()[
            predicted_class_id].item()

        return pd.Series((predicted_class, predicted_probability))


# data_dmatrix = xgb.DMatrix(data=X,label=y)

# # Train the XGBoost model
# xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

# params = {
#     'objective':'reg:squarederror',
#     'colsample_bytree': 0.3,
#     'learning_rate': 0.1,
#     'max_depth': 5,
#     'alpha': 10
# }
