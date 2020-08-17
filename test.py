from keras.models import load_model
from keras_preprocessing import image
import numpy as np

model = load_model('SCC.hdf5')
test_image = image.load_img('C:/Users/hrith/PycharmProjects/SCC3/testing model/Negative/divya.jpeg', target_size=(32,32))
test_image= image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prediction = model.predict(test_image)
print(prediction)
print(prediction.shape)
if(prediction[0][0] == 1):
    print("Basal Cell Carcinoma,NON MELANOMA SKIN CANCER, POSITIVE")
elif (prediction[0][1] == 1):
    print("Benign,NEGATIVE")
elif (prediction[0][2] == 1):
    print("Melanoma,POSITIVE")
else:
    print("Squamous cell carcinoma,NON MELANOMA SKIN CANCER,POSITVE")









