import json

fname = "../data/gy_dpo_1211_judged_final.jsonl"
output = "../data/gy_dpo_1211_winner.jsonl"
output_gpt4 = "../data/gy_dpo_1211_winnergpt4.jsonl"
output_model = "../data/gy_dpo_1211_winnermodel.jsonl"
output_tie = "../data/gy_dpo_1211_tie.jsonl"

with open(fname, 'r', encoding='utf-8') as f:
    lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]


i = 0
data_list = []

for line in lines:
    i += 1
    if line["g1_winner"] not in ["model_1", "model_2", "tie"] or\
            line["g2_winner"] not in ["model_1", "model_2", "tie"]:
        print(line["g1_winner"], line["g2_winner"])
    else:
        data_list.append(line)

d_list = []
with open(output, "w") as fw:
    total_count = len(data_list)
    score = 0
    tie_count = 0
    for line in data_list:
        # traverse df row by row
        if line["g1_winner"] == "tie" and  line["g2_winner"] == "tie":
            line["winner"] = "tie"
            score += 0.5
            tie_count += 1
        elif line["g1_winner"] == "tie":
            line["winner"] = "tie_" + line[line["g2_winner"]]
            score += 0.5
            tie_count += 1
        elif line["g2_winner"] == "tie":
            line["winner"] = "tie_" + line[line["g1_winner"]]
            score += 0.5
            tie_count += 1
        elif line["g1_winner"] != line["g2_winner"]:
            line["winner"] = "tie"
            score += 0.5
            tie_count += 1
        else:
            if line["g1_winner"] == "model_1":
                line["winner"] = line["model_1"]
            else:
                line["winner"] = line["model_2"]
                score += 1
        fw.write(json.dumps(line) + "\n")
        fw.flush()
        d_list.append(line)
        # add win rate
    win_rate = score / total_count
    print("total case: %s, tie case: %s, win rate: %s"%(total_count, tie_count, win_rate))


with open(output_gpt4, "w") as fw:
    for d in d_list:
        if d["winner"] in ["gpt4", "tie_gpt4"]:
            fw.write(json.dumps(d)+"\n")

with open(output_model, "w") as fw:
    for d in d_list:
        if d["winner"] in ["nbmodel", "tie_nbmodel"]:
            fw.write(json.dumps(d)+"\n")

with open(output_tie, "w") as fw:
    for d in d_list:
        if d["winner"] in ["tie"]:
            temp = {
                "g1_winner": d["g1_winner"],
                "g2_winner": d["g2_winner"],
                "winner": d["winner"]
            }
            fw.write(json.dumps(temp)+"\n")