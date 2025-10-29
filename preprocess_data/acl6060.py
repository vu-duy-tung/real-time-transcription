import pandas as pd
import os
import json


df = pd.read_parquet('/data/mino/acl-6060/data/eval-00000-of-00001.parquet')
print("ACL6060 dataset loaded")
print("Column names:", df.columns.tolist())

save_dir = 'save_dir/data/'
os.makedirs(save_dir, exist_ok=True)

audio_save_dir = os.path.join(save_dir, 'acl6060', 'audios')
os.makedirs(audio_save_dir, exist_ok=True)

data_to_export = []

for index, row in df.iterrows():
    audio_file_path = os.path.join(audio_save_dir, row['audio']['path'])
    audio_data = row['audio']['bytes']
    with open(audio_file_path, 'wb') as audio_file:
        audio_file.write(audio_data)

    entry = {'audio_path': os.path.abspath(audio_file_path)}
    entry.update({col: row[col] for col in df.columns if col != 'audio'})
    data_to_export.append(entry)

json_file_path = os.path.join(save_dir, 'acl6060', 'transcriptions.json')
with open(json_file_path, 'w') as json_file:
    json.dump(data_to_export, json_file, indent=4)

import code; code.interact(local=locals())

print("Done")