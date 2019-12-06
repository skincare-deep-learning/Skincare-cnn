import os
import shutil

#path = 'training_set/nevus/'
path = 'test_set/'

folders = os.listdir(path)

#savePath = 'nevus'

source = '_test_set'

for j in range(len(folders)):

    savePath = folders[j]

    if not os.path.exists(savePath + source):
        os.makedirs(savePath + source)

    files = os.listdir(path + savePath + '/')
    files.sort()

    for i in range(len(files)):
        if i > 199:
            shutil.move(path + savePath + '/' + files[i] , savePath + source)