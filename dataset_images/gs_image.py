import json
import csv
from io import BytesIO
import pandas as pd
from isic_api import ISICApi
from io import StringIO
from PIL import Image, ImageDraw, ImageFont
import os
from pandas.io.json import json_normalize
import urllib
import os

#9841 - 9920
## ISIC_557 - 9867

api = ISICApi()
savePath = 'dataset_images/'

if not os.path.exists(savePath):
    os.makedirs(savePath)



data = pd.read_csv('metadata.csv') 
id_data = pd.read_csv('name_id.csv')

diseases = set()


for i in range(len(data)):
    diseases.add(data.loc[i]['diagnosis'])

# Create folders diseases
for diagnosis in diseases:
    folders = str(diagnosis)
    if not os.path.exists(savePath + folders):
        os.makedirs(savePath + folders)

#imageList = api.getJson('image?limit=25000&offset=0&sort=name')

print('Downloading %s images' % len(data))

"""
# Download images from Api code
imageDetails = []
for image in imageList:
    folder = str(data.loc[i]['diagnosis'])
    print(i)
    i+=1
    imageFileResp = api.get('image/%s/download' % id_data.loc[i]['_id'])
    imageFileResp.raise_for_status()
    imageFileOutputPath = os.path.join(savePath + folder, '%s.jpg' % data.loc[i]['name'])
    with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
        for chunk in imageFileResp:
            imageFileOutputStream.write(chunk)
"""

#Download images and compress from Api 

for i in range(len(data)):
    folder = str(data.loc[i]['diagnosis'])
    print(i)

    #if(i < 13840): continue

    imageFileResp = api.get('image/%s/download' % id_data.loc[i]['_id'])
    imageFileResp.raise_for_status()
    im = Image.open(BytesIO(imageFileResp.content))
    im = im.convert('RGB')
    imageFileOutputPath = os.path.join(savePath + folder, '%s.jpg' % data.loc[i]['isic_id'])   
    im.save(imageFileOutputPath ,optimize=True,quality=50) 