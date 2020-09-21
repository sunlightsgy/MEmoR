import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import torch
from base import BaseDataLoader
from utils import read_json
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import AudioFeatureExtractor, TextFeatureExtractor, VisualFeatureExtractor, PersonalityFeatureExtractor

EMOTIONS = ["neutral","joy","anger","disgust","sadness","surprise","fear","anticipation","trust","serenity","interest","annoyance","boredom","distraction"]

class MEmoRDataset(data.Dataset):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        annos = read_json(config['anno_file'])[config['emo_type']]
        # ids = []
        # tmp_annos = []
        # with open(config['id_file']) as fin:
        #     for line in fin.readlines():
        #         ids.append(int(line.strip()))
        
        # for jj, anno in enumerate(annos):
        #     if jj in ids:
        #         tmp_annos.append(anno)
        # annos = tmp_annos
            
        emo_num = 9 if config['emo_type'] == 'primary' else 14
        self.emotion_classes = EMOTIONS[:emo_num]
        
        data = read_json(config['data_file'])
        self.visual_features, self.audio_features, self.text_features = [], [], []
        self.visual_valids, self.audio_valids, self.text_valids = [], [], []
        self.labels = []
        self.charcaters_seq = []
        self.time_seq = []
        self.target_loc = []
        self.seg_len = [] 
        self.n_character = []
        vfe = VisualFeatureExtractor(config)
        afe = AudioFeatureExtractor(config)
        tfe = TextFeatureExtractor(config)
        pfe = PersonalityFeatureExtractor(config)
        self.personality_list = pfe.get_features()
        self.personality_features = []
        

        for jj, anno in enumerate(tqdm(annos)):
            clip = anno['clip']
            target_character = anno['character']
            target_moment = anno['moment']
            on_characters = data[clip]['on_character']
            if target_character not in on_characters:
                on_characters.append(target_character)
            on_characters = sorted(on_characters)
            
            charcaters_seq, time_seq, target_loc, personality_seq = [], [], [], []
            
            for character in on_characters:
                for ii in range(len(data[clip]['seg_start'])):
                    charcaters_seq.append([0 if character != i else 1 for i in range(len(config['speakers']))])
                    time_seq.append(ii)
                    personality_seq.append(self.personality_list[character])
                    if character == target_character and data[clip]['seg_start'][ii] <= target_moment < data[clip]['seg_end'][ii]:
                        target_loc.append(1)
                    else:
                        target_loc.append(0)
            
            vf, v_valid = vfe.get_feature(anno['clip'], target_character)
            af, a_valid = afe.get_feature(anno['clip'], target_character)
            tf, t_valid = tfe.get_feature(anno['clip'], target_character)
            
            
            self.n_character.append(len(on_characters))
            self.seg_len.append(len(data[clip]['seg_start']))
    
            self.personality_features.append(torch.stack(personality_seq))
            self.charcaters_seq.append(torch.tensor(charcaters_seq))
            self.time_seq.append(torch.tensor(time_seq))
            self.target_loc.append(torch.tensor(target_loc, dtype=torch.int8))
            self.visual_features.append(vf)
            self.audio_features.append(af)
            self.text_features.append(tf)
            self.visual_valids.append(v_valid)
            self.audio_valids.append(a_valid)
            self.text_valids.append(t_valid)
            self.labels.append(self.emotion_classes.index(anno['emotion']))            
        

    def __getitem__(self, index):
        
        return torch.tensor([self.labels[index]]), \
            self.visual_features[index], \
            self.audio_features[index], \
            self.text_features[index], \
            self.personality_features[index], \
            self.visual_valids[index], \
            self.audio_valids[index], \
            self.text_valids[index], \
            self.target_loc[index], \
            torch.tensor([1]*len(self.time_seq[index]), dtype=torch.int8), \
            torch.tensor([self.seg_len[index]], dtype=torch.int8), \
            torch.tensor([self.n_character[index]], dtype=torch.int8)
            

    def __len__(self):
        return len(self.visual_features)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) for i in dat]

    def statistics(self):
        all_emotion = [0] * len(self.emotion_classes)
        for emotion in self.labels:
            all_emotion[emotion] += 1
        return all_emotion


class MEmoRDataLoader(BaseDataLoader):
    def __init__(self, config, training=True):
        data_loader_config = config['data_loader']['args']
        self.seed = data_loader_config['seed']
        self.dataset = MEmoRDataset(config)
        self.emotion_nums = self.dataset.statistics()
        super().__init__(self.dataset, data_loader_config['batch_size'], data_loader_config['shuffle'], data_loader_config['validation_split'], data_loader_config['num_workers'], collate_fn=self.dataset.collate_fn)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        weights_per_class = 1. / torch.tensor(self.emotion_nums, dtype=torch.float)
        weights = [0] * self.n_samples
        for idx in range(self.n_samples):
            if idx in valid_idx:
                weights[idx] = 0.
            else:
                label = self.dataset[idx][0]
                weights[idx] = weights_per_class[label]
        weights = torch.tensor(weights)
        train_sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
        valid_sampler = data.SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
