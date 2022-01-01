# -*- coding: utf-8 -*-

from loguru import logger
from HMM_NER.eval_metrics import Metrics
from HMM_NER.hmm import HMMModel
from HMM_NER.utils import save_model
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
    save_model(hmm_model, config.saved_path)

    # eval
    pred_tag_lists = hmm_model.test(test_word_lists, word2id, tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_o=config.remove_o)
    metrics.report_scores()
    metrics.report_confusion_matrix()


def main():
    logger.info("Load Data...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus(config.train_data)
    test_word_lists, test_tag_lists = build_corpus(config.test_data, make_vocab=config.make_vocab)

    logger.info("正在训练评估HMM模型...")
    hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )


if __name__ == "__main__":
    main()
