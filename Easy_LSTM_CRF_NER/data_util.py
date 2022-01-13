def build_corpus(data_path, make_vocab=True):
    word_lists = []
    tag_lists = []
    with open(data_path, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            line = line[:-1]
            if line == "end":
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
            try:
                word, tag = line.split()
                word_list.append(word)
                tag_list.append(tag)
            except Exception:
                continue

    return word_lists, tag_lists


def build_map(lists):
    maps = {'<pad>': 0, '<unk>': 1}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


if __name__ == '__main__':
    data_path = "data/train.char"
    word_lists, tag_lists = build_corpus(data_path)
    word2id = build_map(word_lists)
    tag2id = build_map(tag_lists)
    print(word2id)
    print(tag2id)
