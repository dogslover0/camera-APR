from models.base_arch import Base_arch
from models.ms_apr import MS_APR
import json

def get_model(config, model_name, backbone_path):
    if model_name == "base_apr":
        return Base_arch(backbone_path, config)
    elif model_name[0:6] == "ms_apr":
        return MS_APR(backbone_path, config)
    else:
        raise "{} is not supported! ".format(model_name)

# with open("../config/config.json", 'r') as f:
#     config = json.load(f)
# config = {**config['ms-apr'], **config['general']}
# model = get_model(config, 'ms-apr11', 'efficientnet-b0')
# # print(model)
