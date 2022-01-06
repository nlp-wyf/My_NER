import attrdict
import torch


def load_config():
    CONF = {
        'emb_size': 128,
        'hidden_size': 128,
        'batch_size': 64,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 20,
        'print_step': 5,

        # data path
        'train_data': './data/train.char',
        'dev_data': './data/dev.char',
        'test_data': './data/test.char',

        # save path
        'saved_path': './saved_model/bilstm.pkl',

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    }

    CONF = attrdict.AttrDict(CONF)
    return CONF
