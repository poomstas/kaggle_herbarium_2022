'''
Convert the JSON files to CSVs. Assumes that the test_metadata.json and train_metadata.json files are in ./data/ directory.
'''
# %%
import json
import pandas as pd
from tqdm import tqdm

# %% Convert Test JSON data to CSV
json_file = './data/test_metadata.json'
output_file = './data/test_metadata.csv'

f = open(json_file)
data = json.load(f)

image_ids, file_names, licenses = [], [], [] # 3 columns

for item in tqdm(data):
    image_ids.append(item['image_id'])
    file_names.append(item['file_name'])
    licenses.append(item['license'])

df = pd.DataFrame(list(zip(image_ids, file_names, licenses)),
            columns = ['image_id', 'file_name', 'license'])
    
df.to_csv(output_file)

# %% Conert Train JSON data to CSV
json_file = './data/train_metadata.json'
output_file = './data/train_metadata.csv'

f = open(json_file)
data = json.load(f)

# %% Organize 'annotations'
genus_id, institution_id, category_id, image_id = [], [], [], []
for item in tqdm(data['annotations']):
    genus_id.append(item['genus_id'])
    institution_id.append(item['institution_id'])
    category_id.append(item['category_id'])
    image_id.append(item['image_id'])
df_annotations = pd.DataFrame(list(zip(genus_id, institution_id, category_id, image_id)),
                            columns=['genus_id', 'institution_id', 'category_id', 'image_id'])
df_annotations

# %% Organize 'images'
image_id, file_name, licenses = [], [], []
for item in tqdm(data['images']):
    image_id.append(item['image_id'])
    file_name.append(item['file_name'])
    licenses.append(item['license'])
df_images = pd.DataFrame(list(zip(image_id, file_name, licenses)),
                            columns=['image_id', 'file_name', 'license'])
df_images

# %% Organize 'categories'
image_id, file_name, licenses = [], [], []
category_id, scientificName, family, genus, species, authors = [], [], [], [], [], []
for item in tqdm(data['categories']):
    category_id.append(item['category_id'])
    scientificName.append(item['scientificName'])
    family.append(item['family'])
    genus.append(item['genus'])
    species.append(item['species'])
    authors.append(item['authors']) 
df_categories = pd.DataFrame(list(zip(category_id, scientificName, family, genus, species, authors)),
                        columns = ['category_id', 'scientificName', 'family', 'genus', 'species', 'authors'])
df_categories

# %% Organize 'genera'
genus_id, genus = [], []
for item in tqdm(data['genera']):
    genus_id.append(item['genus_id'])
    genus.append(item['genus'])
df_genera = pd.DataFrame(list(zip(genus_id, genus)),
                            columns=['genus_id', 'genus'])
df_genera

# %% Organize 'institutions'
institution_id, collectionCode = [], []
for item in tqdm(data['institutions']):
    institution_id.append(item['institution_id'])
    collectionCode.append(item['collectionCode'])
df_institutions = pd.DataFrame(list(zip(institution_id, collectionCode)),
                            columns=['institution_id', 'collectionCode'])
df_institutions

# %% Organize 'license'
ids, name, url = [], [], []
for item in tqdm(data['license']):
    ids.append(item['id'])
    name.append(item['name'])
    url.append(item['url'])
df_license = pd.DataFrame(list(zip(ids, name, url)),
                            columns=['license', 'license_name', 'license_url'])
df_license

# %% Merge dataframes
df = df_annotations.merge(df_images, on='image_id', how='outer')
df.drop('image_id', inplace=True, axis=1)
df = df.merge(df_categories, on='category_id', how='outer')
df.drop('category_id', inplace=True, axis=1)
df = df.merge(df_genera, on='genus_id', how='outer')
df.drop('genus_id', inplace=True, axis=1)
df = df.merge(df_institutions, on='institution_id', how='outer')
df.drop('institution_id', inplace=True, axis=1)
df = df.merge(df_license, on='license', how='outer')
df.drop('license', inplace=True, axis=1)
df = df.dropna(subset=['file_name']) # Drop rows that have file_name as NaN

df.to_csv(output_file)
df

