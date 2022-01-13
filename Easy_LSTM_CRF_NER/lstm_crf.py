import torch.nn as nn
from torchcrf import CRF


class LSTM_CRF(nn.Module):
    def __init__(self, word2id, tag2id, embedding_dim=128, hidden_dim=256):
        super(LSTM_CRF, self).__init__()

        self.tag_size = len(tag2id)
        self.vocab_size = len(word2id)

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        # CRF
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)

    def get_emissions(self, x):
        # x [B, L]
        embedded = self.word_embeds(x)
        lstm_out, _ = self.lstm(embedded)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self.get_emissions(sentence)
        if not is_test:
            loss = -self.crf(emissions=emissions, tags=tags, mask=mask, reduction='mean')
            return loss
        else:
            preds = self.crf.decode(emissions, mask)
            return preds
