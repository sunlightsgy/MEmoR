import torch
from utils import read_json
from base import BaseFeatureExtractor


class TextFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print("Initializing TextFeatureExtractor...")
        self.feature_file = config["text"]["feature_file"]
        self.feature_dim = config["text"]["feature_dim"]
        self.features = read_json(self.feature_file)
        self.data = read_json(config["data_file"])
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ret = []
        ret_valid = []
        for character in on_characters:
            for ii, speaker in enumerate(speakers):
                if character == speaker:
                    index = "{}+{}".format(clip, ii)
                    ret.append(torch.tensor(self.features[index]))
                    ret_valid.append(1)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)
        ret = torch.stack(ret, dim=0)
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8)
        return ret, ret_valid
