import torch

from LSTM_NER.model.configure import load_config
from LSTM_NER.model.util import get_tensor
from LSTM_NER.utils import load_model, extend_maps, get_tags, format_results
from LSTM_NER.data_process import build_corpus

config = load_config()


def predict_one_sentence(model, word_lists, word2id, tag2id):
    tensorized_sents, lengths = get_tensor(word_lists, word2id)
    tensorized_sents = tensorized_sents.to(config.device)

    model.eval()
    with torch.no_grad():
        batch_tagids = model.test(tensorized_sents, lengths)

    # 将id转化为标注
    pred_tag_lists = []
    id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
    for i, ids in enumerate(batch_tagids):
        tag_list = []
        for j in range(lengths[i]):
            tag_list.append(id2tag[ids[j].item()])
        pred_tag_lists.append(tag_list)

    return pred_tag_lists[0]


def main():
    print("读取数据...")
    input_str = "李辉, 1992年生, 经营一家科技公司。"
    word_list = list(input_str)
    word_lists = [word_list]
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus(config.train_data)

    print("加载bilstm模型...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id)
    model = load_model(config.saved_path)

    # 预测一句话
    pred_tag_list = predict_one_sentence(model, word_lists, bilstm_word2id, bilstm_tag2id)

    tags = ["PER", "ORG"]
    res = []
    for tag in tags:
        rec_tags = get_tags(pred_tag_list, tag)
        results = format_results(rec_tags, input_str, tag)
        res += results
    print(res)

    for char, tag_id in zip(input_str, pred_tag_list):
        print(char + "_" + tag_id + "|", end="")


if __name__ == "__main__":
    main()
