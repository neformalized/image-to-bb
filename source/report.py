import tensorflow.keras.models.load_model
import matplotlib.pyplot
import hood.dataset
import cv2, numpy
'''
for i in range(len(validation.x)):
    
    result = model.predict(numpy.expand_dims(validation.x[i], axis = 0), verbose = 0)

    img  = (validation.x[i] * 255).astype(numpy.uint8)
    
    cv2.rectangle(img, (int(shape[0] * validation.y[i][0]), int(shape[1] * validation.y[i][1])), (int(shape[0] * validation.y[i][2]), int(shape[1] * validation.y[i][3])), (0, 255, 0), 1)
    cv2.rectangle(img, (int(shape[0] * result[0][0]), int(shape[1] * result[0][1])), (int(shape[0] * result[0][2]), int(shape[1] * result[0][3])), (255, 255, 0), 1)
    
    plt.figure(figsize=(3, 2))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
#
'''