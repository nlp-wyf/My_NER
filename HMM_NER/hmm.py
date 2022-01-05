import torch
from loguru import logger
from tqdm import tqdm


class HMMModel:
    def __init__(self, N, M):
        """
        :param N: 状态数，这里对应存在的标注的种类
        :param M: 观测数，这里对应有多少不同的字
        """
        self.N = N
        self.M = M

        # 状态转移概率矩阵 A[i][j]表示从i状态转移到j状态的概率
        self.A = torch.zeros(N, N)
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        self.B = torch.zeros(N, M)
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        self.Pi = torch.zeros(N)

    def estimate_initial_state_probs(self, tag_lists, tag2id):
        # 估计初始状态概率
        for tag_list in tqdm(tag_lists):
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        # 原因同上
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def estimate_transition_probs(self, tag_lists, tag2id):
        # 估计状态转移概率矩阵, 也就是bigram二元模型
        # estimate p( Y_t+1 | Y_t )
        for tag_list in tqdm(tag_lists):
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i + 1]]
                self.A[current_tagid][next_tagid] += 1
        # 一个重要的问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

    def estimate_emission_probs(self, word_lists, tag_lists, word2id, tag2id):
        # 发射矩阵(观测概率矩阵)参数的估计
        # estimate p(Observation | Hidden_state)
        for tag_list, word_list in tqdm(zip(tag_lists, word_lists), total=len(word_lists)):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        # 原因同上
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        """
        HMM的训练，即根据训练语料对模型参数进行估计,
        因为我们有观测序列以及其对应的状态序列，所以我们
        可以使用极大似然估计的方法来估计隐马尔可夫模型的参数

        :param word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
        :param tag_lists:  列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
        :param word2id: 将字映射为ID
        :param tag2id: 字典，将标注映射为ID
        :return:
        """
        logger.info("Start Train...")
        assert len(tag_lists) == len(word_lists)
        # 估计转移概率矩阵, 发射概率矩阵和初始概率矩阵的参数
        logger.info('estimate initial state matrix')
        self.estimate_initial_state_probs(tag_lists, tag2id)
        logger.info('estimate transition matrix')
        self.estimate_transition_probs(tag_lists, tag2id)
        logger.info('estimate emission matrix')
        self.estimate_emission_probs(word_lists, tag_lists, word2id, tag2id)

        logger.info("Train DONE!")

    def get_p_Obs_State(self, word_id, B):
        # 计算p( observation | state)
        # 如果当前字属于未知, 则讲p( observation | state)设为均匀分布

        Bt = B.t()  # Bt.shape [M, N]
        if word_id is None:
            # 如果字不在字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            # Bt[word_id]表示字为word_id的时候，对应各个标记的概率
            bt = Bt[word_id]
        return bt

    def viterbi_decode(self, word_list, word2id, tag2id):

        # 问题:整条链很长的情况下，非常多的小概率相乘，最后可能造成数值下溢
        # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
        # 同时相乘操作也变成简单的相加操作
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        # 初始化维比特矩阵viterbi, 它的维度为[(标注)状态数, (字列表)观测序列长度]
        # 其中viterbi[i, j]表示标注序列的第j个标注为i的所有单个序列(i_1, i_2, ..i_j)出现的概率最大值
        seq_len = len(word_list)
        viterbi = torch.zeros(self.N, seq_len)
        # backpointer是跟viterbi一样大小的矩阵
        # backpointer[i, j]存储的是标注序列的第j个标注为i时，第j-1个标注的id
        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        backpointer = torch.zeros(self.N, seq_len).long()

        # the first step
        start_wordid = word2id.get(word_list[0], None)
        bt = self.get_p_Obs_State(start_wordid, B)

        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        # 递推公式：
        # viterbi[tag_id, step] = max(viterbi[:, step-1] * self.A.t()[tag_id] * Bt[word])
        # 其中word是step时刻对应的字, self.A.t()[tag_id]表示各个状态转移到tag_id对应的概率
        # 由上述递推公式求后续各步
        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            # bt是在t时刻字为wordid时，状态的概率分布
            bt = self.get_p_Obs_State(wordid, B)
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(viterbi[:, step - 1] + A[:, tag_id], dim=0)
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 回溯，求最优路径
        # 终止，t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(viterbi[:, -1], dim=0)

        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len - 1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list

    def eval(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.viterbi_decode(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

    def get_predict_results(self, text, word2id, tag2id):
        # 预测并打印出预测结果
        # 维特比算法解码
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        best_tag_ids = self.viterbi_decode(text, word2id, tag2id)
        return best_tag_ids
