import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BERT_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768):
        super(BERT_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tag_size = len(tag_to_ix)

        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(embedding_dim, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)

    def _get_features(self, sentence):
        with torch.no_grad():
            embeds, _ = self.bert(sentence)
        enc = self.dropout(embeds)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:  # Training，return loss
            loss = -self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else:  # Testing，return decoding
            decode = self.crf.decode(emissions, mask)
            return decode
