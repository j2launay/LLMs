# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Hate speech dataset"""


import csv
import os
import pandas as pd
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TextClassificationPipeline
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
from datasets import load_dataset, Dataset, DatasetDict

def prepare_hate_dataset():
    data_dir = "./data/"
    all_files_path = os.path.join(data_dir, "all_files")
    texts, labels = [], []
    with open(os.path.join(data_dir, "annotations_metadata.csv"), encoding="utf-8") as csv_file:
        #print(csv_file)
        csv_reader = csv.DictReader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for idx, row in enumerate(csv_reader):
            if idx%50 == 0:
                print(idx)
            text_path = os.path.join(all_files_path, row.pop("file_id") + ".txt")
            with open(text_path, encoding="utf-8") as text_file:
                text = text_file.read()
                #print(text, row)
                texts.append(text)
                labels.append(row['label'])

    # dictionary of lists
    dict_dataset = {'text': texts, 'label': labels}
        
    df = pd.DataFrame(dict_dataset)
        
    # saving the dataframe
    df.to_csv('hate_speech.csv')

def store_dataset(hate, model_name, tokenizer):
    if hate:
        hate_speech = pd.read_csv("./hate_speech.csv")
        hate_speech.drop(["Unnamed: 0"], axis=1, inplace=True)
        hate_speech["label"].replace({"noHate":0, "hate":1, "relation":2}, inplace=True)
        hate_speech.drop(hate_speech[hate_speech['label'] == "idk/skip"].index, inplace = True)
        
        hate_speech_train, hate_speech_test = train_test_split(hate_speech, test_size=0.2, random_state=42)
        hate_speech_train = hate_speech_train.head(5000)
        hate_speech_test = hate_speech_test.head(2000)
        train_dataset = Dataset.from_pandas(hate_speech_train)
        test_dataset = Dataset.from_pandas(hate_speech_test)
        num_labels = 3
        filename_model = "./models/" + model_name + "/hate/"
    else:
        dataset = load_dataset("yelp_review_full")
        test_dataset = dataset["test"]
        num_labels = 5
        filename_model = "./models/" + model_name + "/yelp_review/"
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    if hate:
        small_train_dataset = train_dataset.map(tokenize_function, batched=True)
        small_eval_dataset = test_dataset.map(tokenize_function, batched=True)
    else:
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50))
        small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(50))
        
    try:
        model = AutoModelForSequenceClassification.from_pretrained(filename_model)
    except Exception as e:
        if "Roberta" in filename_model:
            model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target", 
                                                                        num_labels=num_labels, ignore_mismatched_sizes=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
    return model, tokenizer, test_dataset, small_train_dataset, small_eval_dataset, filename_model

def train_model(model, train, test, filename_model):
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model(filename_model)
    
    
if __name__ == '__main__':
    #prepare_hate_dataset()
    hate, model_name = True, "Roberta"
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    model, tokenizer, test_dataset, small_train_dataset, small_eval_dataset, filename_model = store_dataset(hate, model_name, tokenizer)
    #train_model(model, small_train_dataset, small_eval_dataset, filename_model)
    
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    print(pipe("The text to predict", return_all_scores=True))
    print(pipe("The text to predict is a shit", return_all_scores=True))
    #train_model(model, tokenized_texts, hate_speech['labels'])
    