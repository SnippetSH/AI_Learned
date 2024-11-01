import os
from keras.models import load_model, model_from_json

with open('model/lstm_model_structure.json', 'r') as file:
        modelJson = file.read()

model = model_from_json(modelJson)
model.load_weights('model/lstm_model_weights.h5')

model.summary()

