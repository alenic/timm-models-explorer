import timm


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


def get_config(model):
    try:
        config = timm.get_pretrained_cfg(model, allow_unregistered=True)
    except:
        config = None

    return config
