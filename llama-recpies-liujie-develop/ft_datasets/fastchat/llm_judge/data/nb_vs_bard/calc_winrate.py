import pandas as pd
import json

df1 = pd.read_csv("explanations_2023-10-14_fd.csv")

win_num = len(df1[df1["winner"]=="NBLLM"])
loss_num = len(df1[df1["winner"]=="BARD"])
tie_num = len(df1[df1["winner"]=="tie"])

win_rate = (win_num + 0.5 * tie_num) / (win_num + loss_num + tie_num)

print(win_rate)

df2 = pd.read_csv("explanations_2023-11-06_fd.csv")


df2 = df2.sample(30)
print(df2)
df2.to_csv("1106_sample.csv", index=False)


win_num += len(df2[df2["winner"]=="NBLLM"])
loss_num += len(df2[df2["winner"]=="BARD"])
tie_num += len(df2[df2["winner"]=="tie"])

win_rate = (win_num + 0.5 * tie_num) / (win_num + loss_num + tie_num)

print(win_rate)