import pandas as pd
import re

# column names from LIAR dataset
columns = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true",
    "false",
    "half_true",
    "mostly_true",
    "pants_fire",
    "context",
]

# load datasets
train = pd.read_csv("data/train.tsv", sep="\t", names=columns)
valid = pd.read_csv("data/valid.tsv", sep="\t", names=columns)
test = pd.read_csv("data/test.tsv", sep="\t", names=columns)

# combine them
df = pd.concat([train, valid, test])

# label mapping
label_map = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1
}

df["label"] = df["label"].map(label_map)

# keep only needed columns
df = df[["statement", "label"]]


# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


df["statement"] = df["statement"].apply(clean_text)

# save cleaned dataset
df.to_csv("data/clean_data.csv", index=False)

print("Dataset processed successfully")
print(df.head())
