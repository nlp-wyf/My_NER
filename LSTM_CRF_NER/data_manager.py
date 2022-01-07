import copy
import pickle


class DataManager:
    def __init__(self, max_length=100, batch_size=20, data_type='train', tags=[]):
        self.tags = []
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        self.vocab = {"unk": 0}
        self.tag_map = {"O": 0, "START": 1, "STOP": 2}

        if data_type == "train":
            assert tags, Exception("请指定需要训练的tag类型，如[\"ORG\", \"PER\"]")
            self.generate_tags(tags)
            self.data_path = "data/train.char"
        elif data_type == "dev":
            self.data_path = "data/dev.char"
            self.load_data_map()
        elif data_type == "test":
            self.data_path = "data/test.char"
            self.load_data_map()

        self.load_data()
        self.prepare_batch()

    def generate_tags(self, tags):
        for tag in tags:
            for prefix in ["B-", "I-", "E-"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    def load_data_map(self):
        with open("models/data.pkl", "rb") as f:
            self.data_map = pickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    def load_data(self):
        sentence = []
        target = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line[:-1]
                if line == "end":
                    self.data.append([sentence, target])
                    sentence = []
                    target = []
                    continue
                try:
                    word, tag = line.split()
                except Exception:
                    continue
                if word not in self.vocab and self.data_type == "train":
                    self.vocab[word] = max(self.vocab.values()) + 1
                if tag not in self.tag_map and self.data_type == "train" and tag in self.tags:
                    self.tag_map[tag] = len(self.tag_map.keys())
                sentence.append(self.vocab.get(word, 0))
                target.append(self.tag_map.get(tag, 0))
        self.input_size = len(self.vocab.values())
        print("{} data: {}".format(self.data_type, len(self.data)))
        print("vocab size: {}".format(self.input_size))
        print("unique tag: {}".format(len(self.tag_map.values())))
        print("-" * 50)

    def prepare_batch(self):
        index = 0
        while True:
            if index + self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index + self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    @staticmethod
    def pad_data(data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])
        for i in c_data:
            i.append(len(i[0]))
            i[0] = i[0] + (max_length - len(i[0])) * [0]
            i[1] = i[1] + (max_length - len(i[1])) * [0]
        return c_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data) - 1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
