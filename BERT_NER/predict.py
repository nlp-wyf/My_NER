import torch
from transformers import BertTokenizer
from model import Bert_BiLSTM_CRF
from data_loader import idx2tag, tag2idx


def predict(model, sentence, mask):
    model.load_state_dict(torch.load('./ckpt/model.pt'))
    model.eval()
    emission = model._get_features(sentence)
    decode = model.crf.decode(emission, mask)
    return decode


def run():
    MAX_LEN = 30
    model_name = 'E:/huggingface_models/bert-base-chinese/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = Bert_BiLSTM_CRF(tag2idx).to(device)

    input_str = "患者9月余前因反复黑便伴腹部不适于我院消化内科住院，"
    word_list = list(input_str)
    word_list = ["[CLS]"] + word_list + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(word_list)
    token_tensors = torch.LongTensor([token_ids + [0] * (MAX_LEN - len(token_ids))]).to(device)
    mask = (token_tensors > 0)
    mask = mask.to(device)

    decode = predict(model, token_tensors, mask)
    result = [idx2tag[idx] for idx in decode[0]]
    result = result[1: -1]  # 去掉头部的[CLS]和尾部的[SEP]符号

    for char, tag_id in zip(input_str, result):
        print(char + "_" + tag_id + "|", end="")


if __name__ == '__main__':
    run()
