import glob, json, re, string


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

# lan = ["en", "fr", "ge", "it", "po", "ru", "es", "gr", "ka"]
lan = ["es", "gr", "ka"]
all_subtask_names = []
all_subtask_ids = []

all = []
for l in lan:
    lan_all_dev = []
    filenames = glob.glob('../original_data_v4/data/{}/test-articles-subtask-3/*.txt'.format(l))
    with open("../test-articles-subtask-3-{}.txt".format(l), 'w') as outf:
        for file in filenames:
            tmp = {}
            id = file.split('/')[-1].split('article')[-1].split('.')[0]
            tmp["id"] = id
            with open(file, 'r') as inf:
                f = inf.readlines()
                for i in range(len(f)):
                    if len(f[i].strip()) > 0:
                        tmp["para_id"] = i+1
                        tmp['text'] = clean_text(f[i].strip())
                        outf.write(str(tmp))
                        outf.write("\n")
    #                     lan_all_dev.append(tmp)
    #                     print(tmp)
# print(all)
# json.dump(all, outf)
