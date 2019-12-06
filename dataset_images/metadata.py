import json
import csv
import pandas as pd
from isic_api import ISICApi
from pandas.io.json import json_normalize

# Initialize the API; no login is necessary for public data
api = ISICApi(username="SkinCare", password="unbdeeplearning")
outputFileName = 'imagedata'

imageList = api.getJson('image?limit=25000&offset=0&sort=name')

print('Fetching metadata for %s images' % len(imageList))
imageDetails = []
i = 0
for image in imageList:
    print(' ', image['name'])
    # Pull image details
    imageDetail = api.getJson('image/%s' % image['_id'])
    imageDetails.append(imageDetail)

"""
# Testing Parameters 
print("****************************")
print(imageDetails[0]['meta']['clinical']['anatom_site_general'])
print("****************************")
data = json_normalize(imageDetails[0])
print(data.loc[0])

data = json_normalize(imageDetails[0])
print(data.loc[0])
print("========================================================")
print(data.loc[0]['dataset.name'])
"""

# Determine the union of all image metadata fields
metadataFields = set(
    field
    for imageDetail in imageDetails
    for field in imageDetail['meta']['clinical'].keys()
)

metadataFields = ['isic_id'] + sorted(metadataFields)

# print(metadataFields)

outputFilePath = './metadata.csv'

# Write Metadata to a CSV

print('Writing metadata to CSV: %s' % 'metadata.csv')
with open(outputFilePath, 'w') as outputStream:
    csvWriter = csv.DictWriter(outputStream, fieldnames=metadataFields)
    csvWriter.writeheader() # Columns Names
    for imageDetail in imageDetails:
        rowDict = imageDetail['meta']['clinical'].copy()
        rowDict['isic_id'] = imageDetail['name']
        # rowDict['anatom_site_general'] = imageDetail['meta']['clinical']['anatom_site_general'] # Subjective
        csvWriter.writerow(rowDict)