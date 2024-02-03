import timm

def get_models_from_modeltype(model_type):
    models_list = list(timm.models._registry._module_to_models[model_type])
    return models_list


def get_all_modeltypes():
    return list(timm.models._registry._module_to_models.keys())


def get_modeltype_from_model(model):
    model_type = timm.models._registry._model_to_module[model]
    return model_type

def strlist_in_str(name, name_list):
    for n in name_list:
        if n in name: return True
    return False
