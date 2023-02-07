import json, os, glob, re, string
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


lan = ["en", "fr", "ge", "it", "po", "ru"]

all_subtask_names = []
all_dev_names = []
all_subtask_labels = []
all_dev_labels = []
all_subtask_ids = []
all_dev_ids = []
for l in lan:
    filenames = glob.glob('../original_data_v4/data/{}/train-articles-subtask-2/*.txt'.format(l))
    dev_filenames = glob.glob('../original_data_v4/data/{}/dev-articles-subtask-2/*.txt'.format(l))
    # print(len(filenames))
    all_subtask_names.extend(filenames)
    all_dev_names.extend(dev_filenames)

    labels = glob.glob('../original_data_v4/data/{}/train-labels-subtask-2.txt'.format(l))
    dev_labels = glob.glob('../original_data_v4/data/{}/dev-labels-subtask-2.txt'.format(l))
    with open(labels[0], 'r') as label_file:
        f = label_file.readlines()
        for line in f:
            line = line.strip().split()
            ids = line[0]
            labels = line[1]
            all_subtask_ids.append(ids)
            all_subtask_labels.append(labels.split(','))

    with open(dev_labels[0],'r') as label_file2:
        f2 = label_file2.readlines()
        for line in f2:
            line = line.strip().split()
            ids = line[0]
            labels = line[1]
            all_dev_ids.append(ids)
            all_dev_labels.append(labels.split(','))


print(len(all_subtask_labels))
print(len(all_subtask_ids))
print(len(all_subtask_names))
print(len(all_dev_labels))
print(len(all_dev_ids))
print(len(all_dev_names))

## task 2 only
uni_label = set()
for line in all_subtask_labels:
    for label in line:
        uni_label.add(label)
uni_label = list(uni_label)
# print(uni_label)
mlb = MultiLabelBinarizer(classes=uni_label)
# # print([name.split('/')[-1].split('article')[-1].split('.')[0] for name in all_subtask_names])
#
# ## double check if filename match ids
#
output = []
for name in all_subtask_names:
    tmp = {}
    for i, id in enumerate(all_subtask_ids):
        if name.split('/')[-1].split('article')[-1].split('.')[0] == id:
            tmp["id"] = id
            labels = all_subtask_labels[i]
            # if labels[0] == "reporting":
            #     labels = 0
            # elif labels[0] == "opinion":
            #     labels = 1
            # else:
            #     labels = 2
            uni_labels = mlb.fit_transform([labels])

            tmp["label"] = uni_labels[0].tolist()
            # tmp['label'] = labels
            with open(name, 'r') as inf:
                f = inf.readlines()
                cleaned = []
                for line in f:
                    cleaned.append("".join(line.strip()))
                cleaned = list(filter(None, cleaned))
            tmp["cleaned_text"] = clean_text(" ".join(cleaned))
    output.append(tmp)

for name in all_dev_names:
    tmp = {}
    for i, id in enumerate(all_dev_ids):
        if name.split('/')[-1].split('article')[-1].split('.')[0] == id:
            tmp["id"] = id
            labels = all_dev_labels[i]
            # if labels[0] == "reporting":
            #     labels = 0
            # elif labels[0] == "opinion":
            #     labels = 1
            # else:
            #     labels = 2
            uni_labels = mlb.fit_transform([labels])

            tmp["label"] = uni_labels[0].tolist()
            # tmp['label'] = labels
            with open(name, 'r') as inf:
                f = inf.readlines()
                cleaned = []
                for line in f:
                    cleaned.append("".join(line.strip()))
                cleaned = list(filter(None, cleaned))
            tmp["cleaned_text"] = clean_text(" ".join(cleaned))
            # print(tmp)
    output.append(tmp)

print(output)
print(len(output))
#
with open("../train-dev-articles-subtask-2.json", 'w') as ouf:
    json.dump(output, ouf)


with open("../train-articles-subtask-2-label2id.txt", 'w') as ouf2:
    for l in mlb.classes_:
        ouf2.write(l)
        ouf2.write("\n")
