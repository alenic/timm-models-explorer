import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def get_models_from_model_modules(model_type):
    models_list = list(timm.models._registry._module_to_models[model_type])
    models_list = sorted(models_list)
    return models_list


def get_all_model_modules():
    return list(timm.models._registry._module_to_models.keys())


def get_model_module_from_model(model):
    model_module = timm.models._registry._model_to_module[model]
    return model_module


def strlist_in_str(name, name_list):
    for n in name_list:
        if n in name:
            return True
    return False

def get_transforms(model_name):
    pretrained_cfg = timm.get_pretrained_cfg(model_name, allow_unregistered=False).to_dict()
    train_tr = create_transform(**resolve_data_config(pretrained_cfg, pretrained_cfg=pretrained_cfg), is_training=True)
    val_tr = create_transform(**resolve_data_config(pretrained_cfg, pretrained_cfg=pretrained_cfg), is_training=False)
    return str(train_tr), str(val_tr)

def get_config(model):
    try:
        config = timm.get_pretrained_cfg(model, allow_unregistered=True)
    except:
        config = None

    return config
