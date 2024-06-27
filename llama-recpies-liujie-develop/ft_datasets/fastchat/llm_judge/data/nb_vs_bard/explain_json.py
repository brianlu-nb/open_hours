import json

import pandas as pd

df=pd.read_csv("12-03_entity.csv")

all_dict = {}
with open("1203_entity.jsonl", "w") as fw:

    for i in range(len(df)):
        query = df.loc[i, "query"]
        model1_ans = df.loc[i, "model 1 answer"]
        model2_ans = df.loc[i, "model 2 answer"]
        judge = df.loc[i, "judgment"]
        fjudge = df.loc[i, "flipped judgment"]
        winner = df.loc[i, "winner"]

        if winner == "BARD":
            temp = {
                "query": query,
                "model1_ans": model1_ans,
                "model2_ans": model2_ans,
                "judge": judge,
                "fjudge": fjudge
            }
            fw.write(json.dumps(temp, indent=4)+"\n")
        # all_dict[i] = temp

# print(all_dict.pre)

# with open("1203_entity.jsonl", "w") as fw:
#     json.dump(all_dict, fw)

