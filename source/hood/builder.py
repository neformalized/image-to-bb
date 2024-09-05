from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

def create(input_shape, output_shape):
    
    model = Sequential()
    
    ###
    
    model.add(Input(shape=(input_shape[0], input_shape[1], 3)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(output_shape, activation='sigmoid'))
    
    ###
    
    optimizer = Adam(learning_rate = 0.0002)
    model.compile(optimizer = optimizer, loss = "mae")
    
    ###
    
    model.summary()
    
    ###
    
    return model
#