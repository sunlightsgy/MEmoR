import json
import pandas as pd
import importlib
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

from torch.utils.data import dataloader


def find_model_using_name(model_filename, model_name):

    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower():
            model = cls

    if not model:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, model_name))
        exit(0)

    return model

def create_model(config):
    model = find_model_using_name("model.model", config['model']['type'])
    instance = model(config)
    print("model [%s] was created" % (config['model']['type']))
    return instance

def create_dataloader(config):
    dataloader = find_model_using_name("data_loader.data_loaders", config['data_loader']['type'])
    instance = dataloader(config)
    print("dataset [%s] was created" % (config['data_loader']['type']))
    return instance

def create_trainer(model, criterion, metrics, logger, config, data_loader, valid_data_loader):
    trainer = find_model_using_name("trainer.trainer", config['trainer']['type'])
    instance = trainer(model, criterion, metrics,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                    )                  
    print("trainer [%s] was created" % (config['trainer']['type']))
    logger.info(model)
    return instance

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

