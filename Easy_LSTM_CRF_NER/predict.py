import torch
from lstm_crf import LSTM_CRF
from data_loader import NerDataSet


def predict(model, sentence, mask):
    model.load_state_dict(torch.load('./ckpt/bilstm_crf.pth'))
    model.eval()
    with torch.no_grad():
        emission = model.get_emissions(sentence)
        decode = model.crf.decode(emission, mask)
    return decode


def run():
    MAX_LEN = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = "./data/train.char"
    train_dataset = NerDataSet(vocab_file=train_set, data_path=train_set)

    model = LSTM_CRF(word2id=train_dataset.word2id, tag2id=train_dataset.tag2id).to(device)

    wor2id = train_dataset.word2id
    tag2id = train_dataset.tag2id
    id2tag = {idx: tag for idx, tag in enumerate(tag2id)}

    input_str = "市水务局局长吴秀波，39岁，海利装饰有限公司CEO。"
    word_list = list(input_str)
    token_ids = [wor2id.get(s, 1) for s in word_list]
    token_tensors = torch.LongTensor([token_ids + [0] * (MAX_LEN - len(token_ids))]).to(device)
    mask = (token_tensors > 0)
    mask = mask.to(device)

    preds = predict(model, token_tensors, mask)
    result = [id2tag[idx] for idx in preds[0]]

    for char, tag_id in zip(input_str, result):
        print(char + "_" + tag_id + "|", end="")


if __name__ == '__main__':
    run()
