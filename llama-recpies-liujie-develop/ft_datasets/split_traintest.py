import json
from sklearn.model_selection import StratifiedKFold, KFold

prefix = "../data/"
fname = "gy_dpo_1211_winnerall.jsonl"
train_fname = "gy_dpo_1211_winnerall_train.jsonl"
valid_fname = "gy_dpo_1211_winnerall_valid.jsonl"

with open(prefix + fname, "r") as fr:
    lines = fr.readlines()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1992)

type_list = []
idx_list = []
jline_list = []
for idx, l in enumerate(lines):
    jline = json.loads(l)
    jline_list.append(jline)
    # print(jline.keys())
    idx_list.append(idx)
    type_list.append(jline.get("type", "default"))

for idx, (train_index, valid_index) in enumerate(skf.split(idx_list, type_list)):
    with open(prefix + train_fname, "w") as fw:
        for tidx in train_index:
            fw.write(json.dumps(jline_list[tidx])+"\n")

    with open(prefix + valid_fname, "w") as fw:
        for tidx in valid_index:
            fw.write(json.dumps(jline_list[tidx])+"\n")
    exit()


