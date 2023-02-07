import glob, json, random, re, nltk, spacy, string, torch, gc
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch, evaluate
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


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

model_list = glob.glob("./t3_best_models/t3_fold_*.model")

# lan = ["en", "fr", "ge", "it", "po", "ru", "es", "gr", "ka"]
lan = ["es", "gr", "ka"]
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


for l in lan:
    print("Predicting {}".format(l))
    with open("../test/subtask-3/test-articles-subtask-3-{}.txt".format(l), 'r') as inf2:
        f = inf2.readlines()
        pred_data = []
        for line in f:
            li = eval(line)
            tmp = {}
            tmp['para_id'] = li['para_id']
            tmp['text'] = li['text']
            tmp['id'] = li['id']
            pred_data.append(tmp)
        t3_pred = Dataset.from_list(pred_data)

        pred_set = t3_pred.map(tokenize_function, batched=True)

        pred_set = pred_set.remove_columns(["text"])
        # pred_set = pred_set.remove_columns(["id"])
        # pred_set = pred_set.rename_column("label", "labels")
        pred_set.set_format("torch")
        print(pred_set)
        pred_loader = DataLoader(pred_set, batch_size=128, shuffle=False)

    m1_out = []
    m2_out = []
    m3_out = []

    para_id1 = []
    para_id2 = []
    para_id3 = []

    all_id1 = []
    all_id2 = []
    all_id3 = []

    for idx, model_dir in enumerate(model_list):
        print("processing {}".format(model_dir))

        model = XLMRobertaForSequenceClassification.from_pretrained(model_dir, num_labels=24,
                                                                    problem_type="multi_label_classification")
        model.to(device)
        model.eval()
        test_preds = []
        for batch in pred_loader:

            ids = batch.pop('id')
            para_ids = batch.pop('para_id').cpu().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits

            if idx == 0:
                m1_out.extend(logits.cpu().tolist())
                all_id1.extend(ids)
                para_id1.extend(para_ids)
            elif idx == 1:
                m2_out.extend(logits.cpu().tolist())
                all_id2.extend(ids)
                para_id2.extend(para_ids)
            elif idx == 2:
                m3_out.extend(logits.cpu().tolist())
                all_id3.extend(ids)
                para_id3.extend(para_ids)


    print(m1_out)
    print(m2_out)
    print(m3_out)
    c = 0
    for i in range(len(all_id1)):
        if all_id1[i] == all_id2[i] == all_id3[i] and para_id1[i] == para_id2[i] == para_id3[i]:
            c += 1
    print(c)

    ensemble = np.sum([m1_out, m2_out, m3_out],axis=0)
    ensemble = torch.round(torch.sigmoid(torch.tensor(ensemble))).cpu().tolist()


    label_list = ["Appeal_to_Fear-Prejudice", "Appeal_to_Hypocrisy", "Red_Herring", "Exaggeration-Minimisation",
                  "Loaded_Language", "Whataboutism", "Obfuscation-Vagueness-Confusion", "Flag_Waving", "Consequential_Oversimplification",
                  "Name_Calling-Labeling", "Appeal_to_Values", "None", "Causal_Oversimplification","False_Dilemma-No_Choice",
                  "Slogans", "Appeal_to_Popularity", "Repetition", "Guilt_by_Association", "Doubt", "Conversation_Killer", "Appeal_to_Authority",
                  "Straw_Man", "Appeal_to_Time", "Questioning_the_Reputation"]

    with open("../test/subtask-3/t3_pred_{}.txt".format(l), 'w') as outf:
        for i in range(len(all_id1)):
            outf.write(str(all_id1[i]))
            outf.write("\t")
            outf.write(str(para_id1[i]))
            outf.write("\t")
            tmp = []
            for l in range(len(label_list)):
                if ensemble[i][l] != 0:
                    tmp.append(label_list[l])
            out = ",".join(tmp)
            outf.write(out)
            outf.write("\n")

            # for j in ensemble:
            #     for le, l in enumerate(j):
            #         if l > 0:
            #             tmp.append(label_list[le])
            # print(tmp)




