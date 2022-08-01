# zrodlo
# Plik z kodem klasy źródło
import numpy as np
class Source:
    def __init__(self,source_shape:tuple) :
        self._source_shape:tuple= source_shape
    
    def get_data(self) -> np.ndarray:
        rows, cols, channels = self._source_shape
        return np.random.randint(
            256,
            size=rows * cols * channels, 
            dtype=np.uint8,
        ).reshape(self._source_shape)
        
