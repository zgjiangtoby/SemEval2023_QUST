from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from collections import Counter
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


def compute_weights(label):

    class_sample_count = torch.tensor(
        [(label == t).sum() for t in torch.unique(label, sorted=True)])
    weight = 1. / class_sample_count.float()
    sample_weight = torch.tensor([weight[t] for t in label])

    return sample_weight

def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True)

def converter(label):
    out = []
    for i in label:
        out.append(i)
    out = np.array(out, dtype=np.float32)
    return out


device = "cuda:0" if torch.cuda.is_available() else "cpu"

df = pd.read_json("../train/final/train-dev-articles-subtask-1.json", encoding='ISO-8859-1')

n_fold = 10
K_fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2025)
val_all = []

for fold, (train_index, test_index) in enumerate(K_fold.split(df['cleaned_text'].values, df['label'].values)):
    print(f"Fold {fold+1}:")
    train_fold, test_fold = df.iloc[train_index], df.iloc[test_index]

    train_label = train_fold['label'].values
    test_label = test_fold['label'].values

    train_class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label).tolist()
    test_class_weights = compute_class_weight('balanced', classes=np.unique(test_label), y=test_label).tolist()

    train_weight = compute_weights(torch.tensor(train_label))
    test_weight = compute_weights(torch.tensor(test_label))

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    train_set = Dataset.from_pandas(train_fold)
    val_set = Dataset.from_pandas(test_fold)
    train_datasets = train_set.map(tokenize_function, batched=True)
    train_datasets = train_datasets.remove_columns(["cleaned_text", "__index_level_0__",'nation'])
    train_datasets = train_datasets.remove_columns(["id"])
    train_datasets = train_datasets.rename_column("label", "labels")
    train_datasets.set_format("torch")

    val_datasets = val_set.map(tokenize_function, batched=True)
    val_datasets = val_datasets.remove_columns(["cleaned_text", "__index_level_0__",'nation'])
    val_datasets = val_datasets.remove_columns(["id"])
    val_datasets = val_datasets.rename_column("label", "labels")
    val_datasets.set_format("torch")
    train_sampler = WeightedRandomSampler(weights=train_weight, num_samples=len(train_weight))
    train_loader = DataLoader(train_datasets, batch_size=14, sampler=train_sampler)
    val_sampler = WeightedRandomSampler(weights=test_weight, num_samples=len(test_weight))
    val_loader = DataLoader(val_datasets, batch_size=4, sampler=val_sampler)

    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    scheduler = ReduceLROnPlateau(optimizer, "min")
    num_epochs = 30
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(device))
    val_F1 = 0
    tolerant = 5
    saved_epoch = 0
    model.train()
    fold_val_f1 = []
    for epoch in range(num_epochs):
        total_loss = 0
        print("====== Epoch:{} ======".format(epoch + 1))
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            outputs = model(**batch)
            loss = loss_func(outputs.logits, labels)
            #
            total_loss += loss.item()
            loss.backward()
            if step % 50 == 0:
                print("<<<<<<< Training loss: {} >>>>>>>".format(total_loss/(step+1)))

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        print("<<<<<<<<<<  Start evaluation >>>>>>>>>>")
        model.eval()
        eval_preds = []
        eval_labels = []
        eval_total_loss = 0
        dev_total_loss = 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                labels = batch['labels']
                outputs = model(**batch)
                eval_loss = loss_func(outputs.logits, labels)
                scheduler.step(eval_loss)
                eval_total_loss += eval_loss.item()
                eval_labels.extend(labels.cpu().tolist())
                eval_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())

        report_val = classification_report(eval_labels, eval_preds, output_dict=True,
                                           labels=[0, 1, 2], target_names=["reporting", "opinion", "satire"],
                                           )
        f1_macro = report_val['macro avg']['f1-score']
        print("macro F1 is: ", f1_macro)
        fold_val_f1.append(f1_macro)
        if f1_macro > float(val_F1):
            val_F1 = f1_macro
            saved_model = model
            saved_epoch = epoch
            print("saving model with val_f1 {} at Epoch {}".format(val_F1, saved_epoch + 1))
            saved_model.save_pretrained("./checkpoints/t1/t1_fold_{}.model".format(fold+1))

        if epoch - saved_epoch >= tolerant:
            model.cpu()
            saved_model.cpu()
            del model, saved_model, optimizer, loss_func, labels, train_loader, val_loader, outputs, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            max_f1 = max(fold_val_f1)
            print(fold_val_f1)
            print(max_f1)
            with open("./val_finetune_t1_fold{}.txt".format(fold+1), 'w') as outf:
                outf.write("fold {} got {} macro-f1:".format(fold + 1, max_f1))
            print("Early stopping at epoch {}.".format(epoch + 1))

            break




#
# torch.save(model, './bert_en.model')
# print(train_set['label'][:100])
# # print(tokenized_datasets['id'][1])
#
# training_args = TrainingArguments(output_dir='./checkpoints', learning_rate=2e-5,
#                                   per_device_train_batch_size=3, num_train_epochs=10,
#                                   weight_decay=0.01, no_cuda=False)
#
# trainer = Trainer(model=model, args=training_args, train_dataset=train_set, tokenizer=tokenizer)
#
# trainer.train()
#
# trainer.save_model("./t1.model")