train_image_gen.class_indices
import numpy as np
from keras.preprocessing import image

benign_file = ''
benign_img = image.load_img(benign_file)
benign_img = image.img_to_array(benign_img)
benign_img = benign_img/255
benign_img = benign_img.reshape(1,224,224,3)
prediction_prob = model.predict(benign_img)
# Output prediction
print(f'Probability that image is a Belign is: {prediction_prob} ')



