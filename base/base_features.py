import torch

class BaseFeatureExtractor():
    def __init__(self):
        super().__init__()
        

    def get_feature(self, clip, target_character):
        return NotImplementedError