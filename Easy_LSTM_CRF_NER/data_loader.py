import torch
from torch.utils import data
from data_util import build_corpus, build_map


class NerDataSet(data.Dataset):
    def __init__(self, vocab_file, data_path):
        v_word_lists, v_tag_lists = build_corpus(vocab_file)
        word_lists, tag_lists = build_corpus(data_path)
        self.word2id = build_map(v_word_lists)
        self.tag2id = build_map(v_tag_lists)

        self.sents = [[self.word2id[w] for w in sent] for sent in word_lists]
        self.tags = [[self.tag2id[tag] for tag in tag_ls] for tag_ls in tag_lists]

    def __getitem__(self, index):
        seq_len = len(self.sents[index])
        return self.sents[index], self.tags[index], seq_len

    def __len__(self):
        return len(self.sents)


def PadBatch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask
