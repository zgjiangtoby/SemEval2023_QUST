import json, glob
from sklearn.preprocessing import MultiLabelBinarizer


lan = ["en", "fr", "ge", "it", "po", "ru"]

all_subtask_names = []
all_subtask_labels = []
all_subtask_ids = []
all_dev_names = []
all_dev_labels = []
all_dev_ids = []
for l in lan:
    filenames = glob.glob('../original_data_v4/data/{}/train-articles-subtask-3/*.txt'.format(l))
    dev_filenames = glob.glob("../original_data_v4/data/{}/dev-articles-subtask-3/*.txt".format(l))
    # print(len(filenames))
    all_subtask_names.extend(filenames)
    all_dev_names.extend(dev_filenames)

    labels = glob.glob('../original_data_v4/data/{}/train-labels-subtask-3.txt'.format(l))
    dev_labels = glob.glob('../original_data_v4/data/{}/dev-labels-subtask-3.txt'.format(l))

    with open(labels[0], 'r') as label_file:
        f = label_file.readlines()
        for line in f:
            line = line.strip().split()
            ids = [line[0]] + [line[1]]
            if len(line) >2:
                labels = line[2]
            else:
                labels = 'None'

            all_subtask_ids.append(ids)
            all_subtask_labels.append(labels.split(','))
    with open(dev_labels[0], 'r') as label_file2:
        f2 = label_file2.readlines()
        for line in f2:
            line = line.strip().split()
            ids = [line[0]] + [line[1]]
            if len(line) >2:
                labels = line[2]
            else:
                labels = 'None'

            all_dev_ids.append(ids)
            all_dev_labels.append(labels.split(','))

print(len(all_subtask_labels))
print(len(all_subtask_ids))
print(len(all_subtask_names))
print(len(all_dev_labels))
print(len(all_dev_ids))
print(len(all_dev_names))


uni_label = set()
for line in all_subtask_labels:
    for label in line:
        uni_label.add(label)
uni_label = list(uni_label)

mlb = MultiLabelBinarizer(classes=uni_label)

output = []
for name in all_subtask_names:
    for i, id in enumerate(all_subtask_ids):
        if name.split('/')[-1].split('article')[-1].split('.')[0] == id[0]:
            tmp = {}
            tmp["id"] = id[0]
            tmp["para_id"] = para_id = id[1]
            labels = all_subtask_labels[i]
            uni_labels = mlb.fit_transform([labels])

            tmp["label"] = uni_labels[0].tolist()

            # print(name)
            # text = []
            with open(name, 'r') as inf:
                f = inf.readlines()
                tmp['text'] = f[int(para_id)-1].strip()
            output.append(tmp)

for name in all_dev_names:
    for i, id in enumerate(all_dev_ids):
        if name.split('/')[-1].split('article')[-1].split('.')[0] == id[0]:
            tmp = {}
            tmp["id"] = id[0]
            tmp["para_id"] = para_id = id[1]
            labels = all_dev_labels[i]
            uni_labels = mlb.fit_transform([labels])

            tmp["label"] = uni_labels[0].tolist()

            # print(name)
            # text = []
            with open(name, 'r') as inf:
                f = inf.readlines()
                tmp['text'] = f[int(para_id)-1].strip()

            output.append(tmp)

# print(output)
# print(len(output)) # 26663

# #
with open("../train-dev-articles-subtask-3.json", 'w') as ouf:
    json.dump(output, ouf)
# #
with open("../train-dev-articles-subtask-3-label2id.txt", 'w') as ouf2:
    for l in mlb.classes_:
        ouf2.write(l)
        ouf2.write("\n")
