import attrdict


def load_config():
    CONF = {

        # data path
        'train_data': './data/train.char',
        'dev_data': './data/dev.char',
        'test_data': './data/test.char',

        # save path
        'saved_path': './saved_model/hmm.pkl',

        # some flag
        'make_vocab': False,  # 是否生成词表
        'remove_o': False,  # 是否将O标记移除，只关心实体标记
    }

    CONF = attrdict.AttrDict(CONF)
    return CONF
