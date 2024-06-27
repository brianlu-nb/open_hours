import json

with open("../data/gy_dpo_1211_judged_update.jsonl", "r") as fr:
    lines1 = fr.readlines()

with open("../data/gy_dpo_1211_judged_fix.jsonl", "r") as fr:
    lines2 = fr.readlines()

lines = lines1 + lines2

with open("../data/gy_dpo_1211_judged_final.jsonl", "w") as fw:
    for l in lines:
        fw.write(l.strip()+"\n")


# import json
#
# with open("../data/gy_dpo_1211_winnermodel.jsonl", "r") as fr:
#     lines = fr.readlines()
#
# lines1 = []
# for l in lines:
#     jline = json.loads(l)
#     a, b = jline["answer"], jline["rejected_response"]
#     jline["answer"], jline['rejected_response'] = b, a
#     lines1.append(jline)
#
# with open("../data/gy_dpo_1211_winnergpt4.jsonl", "r") as fr:
#     lines = fr.readlines()
#
# lines2 = []
# for l in lines:
#     jline = json.loads(l)
#     print(jline.keys())
#     lines2.append(jline)
#
# lines = lines1 + lines2
# print(len(lines))
#
# with open("../data/gy_dpo_1211_winnerall.jsonl", "w") as fw:
#     for jline in lines:
#         fw.write(json.dumps(jline)+"\n")