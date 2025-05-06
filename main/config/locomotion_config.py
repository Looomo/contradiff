from typing import Any
import torch
import os
from params_proto.neo_proto import ParamsProto, PrefixProto, Proto
import json
import copy

class Configs:
    # def __init__(self, data_dict):
    #     self._dict = data_dict
    #     for key, value in data_dict.items():
    #         setattr(self, key, value)
    
    autoload = False
    def set_from_dict(data_dict):
        Configs._dict = data_dict
        for key, value in data_dict.items():
            setattr(Configs, key, value)

    def __call__():
        return Configs._dict
    
    def add_extra(key, value):
        Configs._dict[key] = value
        setattr(Configs, key, value)

    def savecfg():
        params = copy.deepcopy(Configs._dict)
        params['env'] = str(params['env'] )
        filepath = os.path.join(Configs.savepath, f'config.json')
        with open(filepath , 'w') as fp:
            json.dump(params, fp, indent=4)

        print(f'Configs is saved to bucket: {filepath}')
    