import pandas as pd
import glob
import os

data_folder = "generated_data"

file_pattern = os.path.join(data_folder, "generated_headlines_*.txt")

dfs = []

for file in glob.glob(file_pattern):
    df = pd.read_csv(file, sep="|", header=None, names=["id", "headline"], dtype=str)
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data["id"] = range(1, len(all_data) + 1)  # Reset the ID column to start from 1

all_data.to_csv("consolidated_data/consolidated.txt", sep="|", index=False)
