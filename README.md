[denim.fig.zip](https://github.com/S-Nikita-N/Flex-box-1/files/6874624/denim.fig.zip) - ссылка на макет(fig) для сравнения с версткой  
https://s-nikita-n.github.io/Flex-box-1/ - ссылка на Github Pages  
# Flex-box-1
 1) Сайт состоит из 3 страниц: главная index.html, страницы каталога catalog.html, и образца страницы карточки товара card.html  
 2) Здесь отсутствует какой либо интерактив, (за исключением двух чекбоксов в card.html) так как страница сверстано для отработки технлогии flex-box  
 3) Ссыла в хедере Woman и любая ссылка View all  ведут на страницу catalog.html  
 4) Любая карточка (ссылка в виде фотографии) ведут на страницу card.html  
 5) Well, thats'it!  



import pandas as pd
import numpy as np
import random
import torch
import transformers
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
# from transformers import TrainingArguments, Trainer
# from datasets import load_metric, Dataset
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
import gc
from transformers import get_scheduler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AdamW


def gini(labels, preds):
    gini = 2* roc_auc_score(labels, preds) - 1
    return gini


from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_length = 512,
            padding = 'max_length',
            truncation = True,
            return_tensors='pt',
        )

        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


class BertClassifier:

    def __init__(self, model_path, tokenizer_path, n_classes=2, epochs=1, model_save_path='/content/bert.pt'):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path=model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)
    
    
    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=2, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=2, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.train_loader) * self.epochs
            )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
    
    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        preds = []
        y_true = []
        progress_bar = tqdm(range(len(self.train_loader)))
        for data in self.train_loader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )

            # preds = torch.argmax(outputs.logits, dim=1)
            logits = (np.e**outputs.logits[:,1] / torch.sum(np.e**outputs.logits, axis = 1)).detach().cpu().tolist()
            preds.extend(logits)
            y_true.extend(targets.detach().cpu().tolist())

            loss = self.loss_fn(outputs.logits, targets)
            losses.append(loss.item())
            # correct_predictions += torch.sum(preds == targets)

            

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            progress_bar.update(1)

        # train_acc = correct_predictions.double() / len(self.train_set)
        train_gini = gini(y_true, preds)
        train_loss = np.mean(losses)
        
        return train_gini, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        preds = []
        y_true = []
        # correct_predictions = 0
        progress_bar = tqdm(range(len(self.valid_loader)))
        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    )

                # preds = torch.argmax(outputs.logits, dim=1)
                logits = (np.e**outputs.logits[:,1] / torch.sum(np.e**outputs.logits, axis = 1)).detach().cpu().tolist()
                preds.extend(logits)
                y_true.extend(targets.detach().cpu().tolist())

                loss = self.loss_fn(outputs.logits, targets)
                losses.append(loss.item())
                # correct_predictions += torch.sum(preds == targets)
                
                progress_bar.update(1)

        # val_acc = correct_predictions.double() / len(self.valid_set)

        val_gini = gini(y_true, preds)
        val_loss = np.mean(losses)

        return val_gini, val_loss
    
    def train(self):
        best_accuracy = 0
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        self.model = torch.load(self.model_save_path)
        
    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
              'text': text,
              'input_ids': encoding['input_ids'].flatten(),
              'attention_mask': encoding['attention_mask'].flatten()
          }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction


classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=2,
        epochs=2,
        model_save_path='/content/bert.pt'
)

classifier.preparation(
        X_train=train_text,
        y_train=train_labels,
        X_valid=valid_text,
        y_valid=valid_labels
    )

classifier.train()
