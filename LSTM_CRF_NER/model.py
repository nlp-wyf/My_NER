import torch
from torch import nn

START_TAG = "START"
STOP_TAG = "STOP"


def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class BiLSTM_CRF(nn.Module):

    def __init__(
            self,
            tag_map=None,
            batch_size=20,
            vocab_size=20,
            hidden_dim=128,
            dropout=1.0,
            embedding_dim=100
    ):
        super(BiLSTM_CRF, self).__init__()
        if tag_map is None:
            tag_map = {"O": 0}
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.tag_size = len(tag_map)
        self.tag_map = tag_map

        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.transitions.data[:, self.tag_map[START_TAG]] = -1000.
        self.transitions.data[self.tag_map[STOP_TAG], :] = -1000.

        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim // 2,
                            num_layers=1, bidirectional=True,
                            batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def __get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(sentence).view(self.batch_size, length, self.embedding_dim)

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)
        return logits

    def real_path_score(self, logits, label):
        """
        Score = Emission_Score + Transition_Score
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])
        :param logits: [L, tag_size]
        :param label:  [L,]
        :return:
        """
        score = torch.zeros(1)
        label = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long), label])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_map[STOP_TAG]]
        return score

    def total_score(self, logits):
        """
        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        :param logits: [L, tag_size]
        :return:
        """
        # 有两个变量，obs和previous。previous存储前面步骤的最终结果。obs表示当前单词的信息。
        previous = torch.full((1, self.tag_size), 0)
        for index in range(len(logits)):
            # 把previous和obs扩展成矩阵, 可以提高计算的效率
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            # [tag_size, tag_size]
            scores = previous + obs + self.transitions
            # [1, tag_size]
            previous = log_sum_exp(scores)
        # [1, tag_size]
        previous = previous + self.transitions[:, self.tag_map[STOP_TAG]]
        # calculate total_scores, total_scores is a scalar
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sentences, tags, length):
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)
        for logit, tag, real_len in zip(logits, tags, length):
            logit = logit[:real_len]
            tag = tag[:real_len]
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit)

        return total_score - real_path_score

    def forward(self, sentences, lengths=None):
        """
        predict the tags of the sentences
        :param sentences: [B, L]
        :param lengths: represent the true length of sentence, the default is sentences.size(-1)
        :return:
        """
        sentences = torch.tensor(sentences, dtype=torch.long)
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        scores = []
        paths = []
        for logit, real_len in zip(logits, lengths):
            logit = logit[:real_len]
            score, path = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def __viterbi_decode(self, logits):
        """

        :param logits: [L, tag_size]
        :return:
        """
        # trellis存储历史最好得分, backpointers存储历史最好得分对应的索引
        trellis = torch.zeros(logits.size())
        backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]

        # 回溯，求最优路径, viterbi为存放最优路径的索引列表
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi
