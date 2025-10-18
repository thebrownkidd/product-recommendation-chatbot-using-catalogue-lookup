import utils
import pandas as pd
import json
from tqdm import tqdm

Data = pd.read_csv('catalogue.csv')

data_to_keep = Data[["uniq_id","title", "description","categories"]]
data_formatted = {}
for row in tqdm(data_to_keep.iterrows(), total=data_to_keep.shape[0]):
    idx = row[1]["uniq_id"]
    title = row[1]['title']
    description = row[1]['description']
    categories = utils.MakeList(row[1]['categories'])
    
    disc_cat = {
        "title": title,
        "description": description,
        "categories": categories
    }
    
    #make one json for all the data in this format
    data_formatted[idx] = disc_cat


with open('disc_cat.json', 'w') as f:
    json.dump(data_formatted, f, indent=4)

