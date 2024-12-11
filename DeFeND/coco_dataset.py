import pandas as pd
import json
import numpy as np
import os

ax = json.load(open('data/coco/annotations/instances_train2014.json')) # add link to json file from COCO official website
ax = ax['annotations']  # annotations hold category information

# create new dataframe to hold information on images and corresponding categories they hold
clm = ['image_id', 'category_id']
df = pd.DataFrame(columns=clm) 

for i in range(len(ax)):
    id = ax[i]['image_id']
    iname = str(id).zfill(12)   # pad with zeros to match image name in COCO folder
    df = df.append( {'image_id' : iname,
                     'category_id' : ax[i]['category_id']}, ignore_index=True)  #add row to dataframe
    
# multiple instances of the same object category in an image are stored as separate rows
# deleting them
df = df.drop_duplicates(df.columns)

# cleaning the dataframe content
img_names = np.unique(df['image_id']) # get set of image names

for i in img_names:
    # if the image contains an element under person category (category #1)
    if len(df[(df['image_id'] == str(i)) & (df['category_id'] == 1)]) != 0:
        # remove rows that indicate presence of other category elements in the image 
        indices = df[(df['image_id'] == str(i)) & (df['category_id'] != 1)].index
        if(len(indices) != 0):
            df = df.drop(index=indices)

# separate into 2 dataframes
df_person = df[df['category_id'] == 1]
df_not = df[df['category_id'] != 1]

# set category_id = 0 to indicate not_person
df_not['category_id'] = 0

# there may be several not_person elements leading to redundant rows, eliminating them
df_not = df_not.drop_duplicates()

path = 'data/'   # path to store the categorized image files
curpath = 'data/coco/train2014'    # path to COCO image files

# make appropriate folders
os.mkdir(path + 'vww')
os.mkdir(path + 'vww/person')
os.mkdir(path + 'vww/notperson')

p = path + 'vww/person/'
n_p = path + 'vww/notperson/'

dfp = list(df_person['image_id'])
dfnp = list(df_not['image_id'])

n = 300 # number of images to pick from each category

for i in range(n):
    os.rename(curpath + str(dfp[i]).zfill(12) + '.jpg', p + str(dfp[i]).zfill(12) + '.jpg')
    os.rename(curpath + str(dfnp[i]).zfill(12) + '.jpg', n_p + str(dfnp[i]).zfill(12) + '.jpg')





