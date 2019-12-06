import csv
import pandas as pd
from isic_api import ISICApi
from pandas.io.json import json_normalize

api = ISICApi(username="SkinCare", password="unbdeeplearning")


imageList = api.getJson('image?limit=25000&offset=0&sort=name') # Take names and id´s

imageList = json_normalize(imageList) # Dataframe Json

imageList.to_csv(index=False) # Convert to CSV  

#print (imageList)

outputFilePath = './name_id.csv'

export_csv = imageList.to_csv (r'./name_id.csv', index = None, header=True) 
