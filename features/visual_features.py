import torch
import os
from base import BaseFeatureExtractor
from utils import read_json


class VisualFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print('Initializing VisualFeatureExtractor...')
        self.speakers = config['speakers']
        self.faces_features_dir = config['visual']['faces_feature_dir']
        self.faces_names_dir = config['visual']['faces_names_dir']
        self.obj_feature_dir = config['visual']['obj_feature_dir']
        self.env_features_dir = config['visual']['env_feature_dir']
        self.data = read_json(config['data_file'])
        self.feature_dim = config['visual']['dim_env'] + config['visual']['dim_obj'] + config['visual']['dim_face']
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        fps = 23.976
        ret = []
        ret_valid = []
        on_characters = self.data[clip]['on_character']
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        seg_start = self.data[clip]['seg_start']
        seg_end = self.data[clip]['seg_end']
        overall_start = self.data[clip]['start']
        with open(os.path.join(self.faces_names_dir, clip+'.txt')) as fin:
            faces_image_names = fin.readline().strip().split('\t')
        
        threshold = 10

        if len(faces_image_names) > threshold:
            face_features = torch.load(os.path.join(self.faces_features_dir, clip+'.pt'), map_location='cpu')
            obj_features = torch.load(os.path.join(self.obj_feature_dir, clip+'.pt'), map_location='cpu')
            env_features = torch.load(os.path.join(self.env_features_dir, clip+'.pt'), map_location='cpu')
            for character in on_characters:
                for ii in range(len(seg_start)):
                    begin_sec, end_sec = seg_start[ii] - overall_start, seg_end[ii] - overall_start
                    begin_idx, end_idx = int(begin_sec * fps), int(end_sec * fps)
                    character_face_feature = []
                    for jj, image_name in enumerate(faces_image_names):
                        idx, person = tuple(image_name[:-4].split('_'))
                        if begin_idx <= int(idx) <= end_idx and person.lower() == self.speakers[character]:
                            character_face_feature.append(face_features[jj])
                    face_num = len(character_face_feature)
                    
                    if face_num > threshold:
                        ret_in = []
                        ret_in.append(torch.mean(torch.stack(character_face_feature), dim=0))
                        ret_in.append(torch.mean(obj_features[begin_idx:end_idx, :], dim=0))
                        ret_in.append(torch.mean(env_features[begin_idx:end_idx, :], dim=0))
                        # print(torch.cat(ret_in).shape)
                        ret.append(torch.cat(ret_in))
                        ret_valid.append(1)
                    else:
                        ret.append(self.missing_tensor) 
                        ret_valid.append(0)                    
            ret = torch.stack(ret, dim=0)
            ret_valid = torch.tensor(ret_valid, dtype=torch.int8)
            return ret, ret_valid
        else:
            return torch.zeros((len(on_characters) * len(seg_start), self.feature_dim)), torch.zeros(len(on_characters) * len(seg_start), dtype=torch.int8)


