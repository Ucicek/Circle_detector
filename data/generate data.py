import numpy as np

from main import generate_examples

generator = generate_examples()

directory = 'path/to/train'

for i in range(0, 3000):
  img,params = next(generator)
  image = np.array(img, dtype =np.float32)
  row = np.array(params[0], dtype=np.int32)
  col = np.array(params[1] , dtype = np.int32)
  radius = np.array(params[2], dtype = np.int32)
  name = directory + "/" + "Circle_train" + str(i)
  np.savez(image=image, x=row, y=col, r=radius, file=name)

directory = 'path/to/validation'
for i in range(0, 3000):
  img, params = next(generator)
  image = np.array(img, dtype =np.float32)
  row = np.array(params[0], dtype=np.int32)
  col = np.array(params[1] , dtype = np.int32)
  radius = np.array(params[2], dtype = np.int32)
  name = directory + "/" + "Circle_validation" + str(i)
  np.savez(image=image, x=row, y=col, r=radius, file=name)

directory = 'path/to/test'
for i in range(0, 3000):
  img,params = next(generator)
  image = np.array(img, dtype =np.float32)
  row = np.array(params[0], dtype=np.int32)
  col = np.array(params[1] , dtype = np.int32)
  radius = np.array(params[2], dtype = np.int32)
  name = directory + "/" + "Circle_test" + str(i)
  np.savez(image=image, x=row, y=col, r=radius, file=name)