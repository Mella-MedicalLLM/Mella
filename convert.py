import pandas as pd

df = pd.read_csv("data.csv")
df = df.filla("")

text_col = []
for _, row in df.iterrows():
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = str(row["instruction"])
    input_query = str(row["input"])
    response = str(row["output"])

    if len(input_query.strip()) == 0:
        text = prompt + "### Instruction:\n" + instruction + "\n### Response:\n" + response

    else:
        text = prompt + "### Instruction:\n" + instruction + "\n### Input:\n" + input_query + "\n### Response:\n" + response

    text_col.append(text)

df.loc[:, "text"] = text_col
print(df.head())

df.to_csv("train.csv", index=False)
