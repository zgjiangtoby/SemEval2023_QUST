import json, re, glob, string

lan = ["en", "fr", "ge", "it", "po", "ru", "es", "gr", "ka"]

all_subtask_names = []
all_subtask_ids = []

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

for l in lan:
    lan_all_dev = []
    filenames = glob.glob('../original_data_v4/data/{}/test-articles-subtask-2/*.txt'.format(l))
    with open("../test-articles-subtask-2-{}.json".format(l), 'w') as outf:
        for file in filenames:
            tmp = {}
            id = file.split('/')[-1].split('article')[-1].split('.')[0]
            tmp["id"] = id
            with open(file, 'r') as inf:
                f = inf.readlines()
                cleaned = []
                for line in f:
                    cleaned.append("".join(line.strip()))
                cleaned = list(filter(None, cleaned))
            tmp["text"] = clean_text(" ".join(cleaned))
            lan_all_dev.append(tmp)
        json.dump(lan_all_dev, outf)



