import yaml

import torch
import torch.optim as optim
from data_manager import DataManager
from model import BiLSTMCRF
from utils import f1_score, get_tags, format_result, save_params, load_params


class ChineseNER(object):

    def __init__(self, entry="train"):
        self.load_config()
        self.__init_model(entry)

    def __init_model(self, entry):
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            save_params(data)
            dev_manager = DataManager(batch_size=30, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()
        elif entry == "predict":
            data_map = load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()

    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen, Loader=None)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 128,
                "hidden_size": 128,
                "batch_size": 64,
                "dropout": 0.5,
                "model_path": "models/",
                "tags": ["ORG", "PER"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(20):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                progress = ("█" * int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                    epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                ))
                self.evaluate()
                print("-" * 50)
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

    def evaluate(self):
        sentences, labels, length = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences)
        print("\teval")
        for tag in self.tags:
            f1_score(labels, paths, tag, self.model.tag_map)

    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
        input_vec = [self.vocab.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1)
        _, paths = self.model(sentences)

        entities = []
        for tag in self.tags:
            tags = get_tags(paths[0], tag, self.tag_map)
            entities += format_result(tags, input_str, tag)
        return entities


if __name__ == "__main__":
    # cn = ChineseNER("train")
    # cn.train()
    cn = ChineseNER("predict")
    print(cn.predict())
    # if len(sys.argv) < 2:
    #     print("menu:\n\ttrain\n\tpredict")
    #     exit()
    # if sys.argv[1] == "train":
    #     cn = ChineseNER("train")
    #     cn.train()
    # elif sys.argv[1] == "predict":
    #     cn = ChineseNER("predict")
    #     print(cn.predict())
