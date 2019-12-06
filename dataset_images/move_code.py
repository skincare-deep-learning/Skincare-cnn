import os
import shutil

#path = 'training_set/nevus/'
path = 'training_set/melanoma/'

files = os.listdir(path)

#savePath = 'nevus'
savePath = 'melanoma'
if not os.path.exists(savePath):
    os.makedirs(savePath)

files.sort()

for i in range(len(files)):
    if i > 1000:
        shutil.move(path + files[i], savePath)


#print (folders[0])