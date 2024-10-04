import pandas as pd
import json
import re

src_file = "src/Frenchdico.jsonl"
words = []
genders = []

df = pd.read_json(src_file, lines= True)
df = df[['word', "head_templates"]]

for index, row in df.iterrows():
    word = row["word"]

    if isinstance(row['head_templates'], list) and len(row['head_templates']) > 0:
        try:
            gender = str(row['head_templates'][0].get("args", {}).get("1"))
            if gender not in ("m", "f"):
                gender = None
        except (IndexError, AttributeError):
            gender = None
    else:
        gender = None

    if word and gender:
        words.append(word)
        genders.append(gender)


clean_df = pd.DataFrame({
    'Word': words,
    'Gender': genders
})

csv_file = 'src/Dico.csv'
clean_df.to_csv(csv_file, index=False)