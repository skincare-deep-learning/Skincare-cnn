# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 

path = os.listdir('training_set/')
path.sort()
#print (path)


datagen = ImageDataGenerator( 
		rotation_range = 40, 
		shear_range = 0.2, 
		zoom_range = 0.2, 
		horizontal_flip = True, 
		brightness_range = (0.5, 1.5)) 
	
quantity = [4, 0, 3, 6, 8]
z = -1
for i in range(len(path)):
#i = 3
	if i < 3: 
		continue
	if i == 4: 
		continue
	root = 'training_set/'
	files = os.listdir( root + path[i] + '/')
	files.sort()
	z +=1
	print (path)
	print (path[i])
	print(quantity[z])
	way = root + path[i] + '/'
	print(len(files))
	for j in range(len(files)):
			img = load_img(way + files[j]) 
			#img = img.convert('RGB')
			img = img.resize((1280,720))


			x = img_to_array(img) 

			x = x.reshape((1, ) + x.shape) 
	# Generating and saving 5 augmented samples 
	# using the above defined parameters. 
			k = 0
			for batch in datagen.flow(x, batch_size = 1, 
							save_to_dir = way, 
							save_prefix = files[j] , save_format ='jpg'): 
				k += 1
				if k > quantity[z]: 
					break
