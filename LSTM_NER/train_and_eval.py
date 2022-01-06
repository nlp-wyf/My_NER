from copy import deepcopy

import torch
import torch.optim as optim

from LSTM_NER.data_process import build_corpus
from LSTM_NER.evaluating import Metrics
from LSTM_NER.model.configure import load_config
from LSTM_NER.model.bilstm import BiLSTM
from LSTM_NER.utils import extend_maps, save_model, load_model
from LSTM_NER.model.util import get_tensor, sort_by_lengths, cal_loss

config = load_config()


def train(model, train_data, dev_data, word2id, tag2id, optimizer):
    word_lists, tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
    dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

    batch_size = config.batch_size
    for e in range(1, config.epochs + 1):
        step = 0
        losses = 0.
        for idx in range(0, len(word_lists), batch_size):
            batch_sents = word_lists[idx:idx + batch_size]
            batch_tags = tag_lists[idx:idx + batch_size]
            step += 1
            losses += train_step(model, batch_sents, batch_tags, word2id, tag2id, optimizer)

            if step % config.print_step == 0:
                total_step = (len(word_lists) // batch_size + 1)
                print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                    e, step, total_step,
                    100. * step / total_step,
                    losses / config.print_step
                ))
                losses = 0.

        # 每轮结束测试在验证集上的性能，保存最好的一个
        val_loss = validate(model, dev_word_lists, dev_tag_lists, word2id, tag2id)
        print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))


def train_step(model, batch_sents, batch_tags, word2id, tag2id, optimizer):
    model.train()
    # 准备数据
    tensorized_sents, lengths = get_tensor(batch_sents, word2id)
    tensorized_sents = tensorized_sents.to(config.device)
    targets, lengths = get_tensor(batch_tags, tag2id)
    targets = targets.to(config.device)
    # forward
    scores = model(tensorized_sents, lengths)

    optimizer.zero_grad()
    loss = cal_loss(scores, targets, tag2id).to(config.device)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, dev_word_lists, dev_tag_lists, word2id, tag2id):
    best_val_loss = 1e18
    model.eval()
    with torch.no_grad():
        val_losses = 0.
        val_step = 0
        for ind in range(0, len(dev_word_lists), config.batch_size):
            val_step += 1
            # 准备batch数据
            batch_sents = dev_word_lists[ind:ind + config.batch_size]
            batch_tags = dev_tag_lists[ind:ind + config.batch_size]
            tensorized_sents, lengths = get_tensor(batch_sents, word2id)
            tensorized_sents = tensorized_sents.to(config.device)
            targets, lengths = get_tensor(batch_tags, tag2id)
            targets = targets.to(config.device)

            # forward
            scores = model(tensorized_sents, lengths)

            # 计算损失
            loss = cal_loss(scores, targets, tag2id).to(config.device)
            val_losses += loss.item()
        val_loss = val_losses / val_step

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("保存模型...")
            best_model = deepcopy(model)
            save_model(best_model, config.saved_path)

        return val_loss


def test(model, word_lists, tag_lists, word2id, tag2id):
    """返回最佳模型在测试集上的预测结果"""
    # 准备数据
    word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
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

    # indices存有根据长度排序后的索引映射的信息
    # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
    ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*ind_maps))
    pred_tag_lists = [pred_tag_lists[i] for i in indices]
    tag_lists = [tag_lists[i] for i in indices]

    return pred_tag_lists, tag_lists


def run(flag="train"):
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus(config.train_data)
    dev_word_lists, dev_tag_lists = build_corpus(config.dev_data, make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus(config.test_data, make_vocab=False)
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id)

    vocab_size = len(bilstm_word2id)
    out_size = len(bilstm_tag2id)

    model = BiLSTM(vocab_size, config.emb_size, config.hidden_size, out_size)
    model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if flag == "train":
        train(model,
              (train_word_lists, train_tag_lists),
              (dev_word_lists, dev_tag_lists),
              word2id, tag2id, optimizer)
    else:
        model = load_model(config.saved_path)
        pred_tag_lists, tag_lists = test(model, test_word_lists, test_tag_lists, word2id, tag2id)
        metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=False)
        metrics.report_scores()
        metrics.report_confusion_matrix()


if __name__ == '__main__':
    run(flag='test')
