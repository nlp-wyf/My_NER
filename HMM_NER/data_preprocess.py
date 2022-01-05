from HMM_NER.configure import load_config

config = load_config()


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

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps
