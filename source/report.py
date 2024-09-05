from tensorflow.keras.models import load_model
from hood import dataset, image, label
import matplotlib.pyplot as plt
import cv2, numpy

model = load_model("/content/model_evaluated.keras")

shape = model.input_shape[1:3]

dataset_path_validation = ["/content/image-to-bb/dataset/validation.txt", "/content/image-to-bb/dataset/validation/"]
dataset_validation = dataset.Dataset(dataset_path_validation)

for item in dataset_validation.data:
    
    x = image.process(item[0], shape)
    y = list(map(float, label.process(item[1])))
    
    result = model.predict(numpy.expand_dims(x, axis = 0), verbose = 0)

    img  = (x * 255).astype(numpy.uint8)
    
    cv2.rectangle(img, (int(shape[0] * y[0]), int(shape[1] * y[1])), (int(shape[0] * y[2]), int(shape[1] * y[3])), (0, 255, 0), 1)
    cv2.rectangle(img, (int(shape[0] * result[0][0]), int(shape[1] * result[0][1])), (int(shape[0] * result[0][2]), int(shape[1] * result[0][3])), (255, 255, 0), 1)
    
    plt.figure(figsize=(3, 2))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
#