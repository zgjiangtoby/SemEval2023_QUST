import glob, json, random, re, nltk, spacy, string, torch, gc
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, get_scheduler
from torch.utils.data import DataLoader

import torch, evaluate
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def clean_text(text):
    # to lower case
    text = text.lower()
    # remove links
    text = re.sub('https:\/\/\S+', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove next line
    text = re.sub(r'[^ \w\.]', '', text)
    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    return text


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_prob(test_preds):
    preds_final = []
    for i in test_preds:
        tmp = []
        for j in i:
            if j < 0:
                j = 0.
                tmp.append(j)
            else:
                j = 1.
                tmp.append(j)
        preds_final.append(tmp)
    return preds_final


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_list = glob.glob("../results/best_models/t1_finetune_10fold*.model")

lan = ["en", "fr", "ge", "it", "po", "ru", "es", "gr", "ka"]

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
#
for l in lan:
    print("Predicting {}".format(l))
    with open("../test/subtask-1/test-articles-subtask-1-{}.json".format(l), 'r') as inf2:
        pred_data = json.load(inf2)
        # for idx, t in enumerate(pred_data):
        #     pred_data[idx]['text'] = clean_text(t['text'])
        t1_pred = Dataset.from_list(pred_data)
        pred_set = t1_pred.map(tokenize_function, batched=True)
        pred_set = pred_set.remove_columns(["text"])
        # pred_set = pred_set.remove_columns(["id"])
        # pred_set = pred_set.rename_column("label", "labels")
        pred_set.set_format("torch")
        print(pred_set)
        pred_loader = DataLoader(pred_set, batch_size=8, shuffle=False)

    m1_out = []
    m2_out = []
    m3_out = []
    all_id1 = []
    all_id2 = []
    all_id3 = []

    for idx, model_dir in enumerate(model_list):
        print("processing {}".format(model_dir))

        saved_model = torch.load(model_dir)
        model = saved_model.to(device)
        model.eval()
        test_preds = []
        for batch in pred_loader:

            ids = batch.pop('id')

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits

            if idx == 0:
                m1_out.extend(logits.cpu().tolist())
                all_id1.extend(ids)
            elif idx == 1:
                m2_out.extend(logits.cpu().tolist())
                all_id2.extend(ids)
            elif idx == 2:
                m3_out.extend(logits.cpu().tolist())
                all_id3.extend(ids)

    print(m1_out)
    print(m2_out)
    print(m3_out)


    c = 0
    for i in range(len(all_id1)):
        if all_id1[i] == all_id2[i] == all_id3[i]:
            c += 1
    print(c)


    ensemble = np.sum([m1_out, m2_out, m3_out],axis=0)
    ensemble = np.argmax(ensemble, axis=-1)
    print(ensemble)


    with open("../test/subtask-1/t1_pred_{}.txt".format(l), 'w') as outf:
        for i in range(len(all_id1)):
            outf.write(str(all_id1[i]))
            outf.write("\t")
            if ensemble[i] == 0:
                outf.write("reporting")
                outf.write("\n")
            elif ensemble[i] == 1:
                outf.write("opinion")
                outf.write("\n")
            else:
                outf.write("satire")
                outf.write("\n")



