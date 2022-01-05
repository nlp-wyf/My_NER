# -*- coding: utf-8 -*-

from loguru import logger
from HMM_NER.eval_metrics import Metrics
from HMM_NER.hmm import HMMModel
from HMM_NER.utils import save_model, load_model, get_tags, format_results
from HMM_NER.data_preprocess import build_corpus
from HMM_NER.configure import load_config

config = load_config()


def hmm_train_eval(train_data, test_data, word2id, tag2id):
    """训练并评估HMM模型"""

    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    # train
    hmm_model = HMMModel(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    logger.info('Save Model!')
    save_model(hmm_model, config.saved_path)

    # eval
    logger.info('Start Evaluate')
    pred_tag_lists = hmm_model.eval(test_word_lists, word2id, tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_o=config.remove_o)
    metrics.report_scores()
    metrics.report_confusion_matrix()


def predict_one_sentence(save_model, input_str, word2id, tag2id):
    logger.info("加载HMM模型...")
    hmm_model = load_model(save_model)
    logger.info("预测")
    my_tags = ["PER", "ORG"]
    best_tag_ids = hmm_model.get_predict_results(input_str, word2id, tag2id)

    entities = []
    for tag in my_tags:
        tags = get_tags(best_tag_ids, tag)
        entities += format_results(tags, input_str, tag)
    print(entities)

    for char, tag_id in zip(input_str, best_tag_ids):
        print(char + "_" + tag_id + "|", end="")


def main(train=True):
    logger.info("Load Data...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus(config.train_data)
    # tag2id {'O': 0, 'B-PER': 1, 'E-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'E-ORG': 5, 'I-PER': 6}
    dev_word_lists, dev_tag_lists = build_corpus(config.dev_data, make_vocab=config.make_vocab)

    if train:
        logger.info("正在训练评估HMM模型...")
        hmm_train_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            word2id,
            tag2id
        )
    else:
        HMM_MODEL_PATH = './saved_model/hmm.pkl'
        input_str = "李华今天去了金辉科技有限公司"
        predict_one_sentence(HMM_MODEL_PATH, input_str, word2id, tag2id)


if __name__ == "__main__":
    # main(train=True)
    main(train=False)
