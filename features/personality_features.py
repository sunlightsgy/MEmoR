import torch
import torch.nn.functional as F
from base import BaseFeatureExtractor

class PersonalityFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        print('Initializing PersonalityFeatureExtractor...')
        self.characters = config['speakers']
        self.features = []
        with open(config['personality']['anno_file']) as fin:
            for ii, line in enumerate(fin.readlines()):
                features = [float(i) for i in line.strip().split(',')]
                self.features.append(features)
        self.features = torch.tensor(self.features)
        self.features = F.normalize(self.features, dim=0)
    
    def get_features(self):
        return self.features


