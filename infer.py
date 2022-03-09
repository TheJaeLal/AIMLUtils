import keras
import numpy as np

class CNNModel:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path, compile=False)
        self.inp_channels = self.model.input.shape.as_list()[-1]
        self.inp_height = self.model.input.shape.as_list()[-1]
        self.inp_width = self.model.input.shape.as_list()[-1]
        return
    
    
    def predict(self, inp, batch=False):
        
        if not batch:
            inp = np.expand_dims(inp, axis=0)
        
        out = self.model.predict(inp)
        
        if not batch:
            out = np.squeeze(out, axis=0)
        
        return out.flatten()
