from . import cc12m_1, wikiart_256
models = {
    'cc12m_1': cc12m_1.CC12M1Model,
    'cc12m_1_cfg': cc12m_1.CC12M1Model,
    'wikiart_256': wikiart_256.WikiArt256Model,

}


def get_model(model):
    return models[model]


def get_models():
    return list(models.keys())
