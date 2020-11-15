#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip install transformers==3.0.2')

from torch import cuda
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
logging.basicConfig(level=logging.ERROR)
plt.style.use('fivethirtyeight')


class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Phrase
        self.targets = self.data.Sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def makeDataSet(path):
  test = pd.read_csv(path,delimiter = ';')
  
  test.rename(columns={'texto':'Phrase'},inplace = True)
  test['Sentiment'] = 0
  Test_df = test[['Phrase','Sentiment']]
  Test_df = Test_df.reset_index(drop=True)
  
  Test_df['Phrase'] = Test_df['Phrase'].apply(lambda x: clean_text(x))
  
  Test_df.to_csv('VerificarCsv.txt',index = False)

  testing_set = SentimentData(Test_df, tokenizer, 256)
  test_params = {'batch_size': 4,
                'shuffle': False,
                'num_workers': 0
                }

  testing_loader = DataLoader(testing_set, **test_params)

  return testing_loader

def predicao(model, testing_loader):
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    predicoes = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):

            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids).squeeze()
            big_val, big_idx = torch.max(outputs.data, dim=1)

            predicoes.append(big_idx.tolist())
    return np.array(predicoes).flatten()

def getPercents(dataFrame):
    positivo = dataFrame[dataFrame.Analysis == 'Positivo']
    pPositivo = round((positivo.shape[0] / dataFrame.shape[0]) * 100, 1)

    negativo = dataFrame[dataFrame.Analysis == 'Negativo']
    pNegativo = round((negativo.shape[0] / dataFrame.shape[0]) * 100, 1)

    neutro = dataFrame[dataFrame.Analysis == 'Neutro']
    pNeutro = round((neutro.shape[0] / dataFrame.shape[0]) * 100, 1)

    return pPositivo, pNegativo, pNeutro
    #return f'positivo: {pPositivo}%, negativo: {pNegativo}%, neutro: {pNeutro}%'

def getAnalysis(score):
    if score == 0:
        return 'Negativo'
    elif score == 2:
        return 'Neutro'
    else:
        return 'Positivo'
    
def graficoSentiment(lista):
    df = pd.DataFrame(lista, columns=['Sentiment'])
    df['Analysis'] = df['Sentiment'].apply(getAnalysis)

    #print(getPercents(df))
    percentPOS, percentNEG, percentNEU = getPercents(df)

    df['Analysis'].value_counts()

    plt.figure(figsize=(10,6))
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    df['Analysis'].value_counts().plot(kind='bar')
    plt.savefig('plot.png', format='png')
    plt.show()

    return percentPOS, percentNEG, percentNEU


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    loaded_model = torch.load("pytorch_roberta_sentiment.bin",map_location=torch.device('cuda'))

    dataSet = makeDataSet('dados.csv')
    predicao = predicao(loaded_model,dataSet)
    percentPOS, percentNEG, percentNEU = graficoSentiment(predicao)

    return percentPOS, percentNEG, percentNEU
