import pickle


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def format_results(result, text, tag):
    entities = []
    for i in result:
        begin, end = i
        entities.append({
            "start": begin,
            "stop": end + 1,
            "word": text[begin:end + 1],
            "type": tag
        })
    return entities


def get_tags(path, tag):
    begin_tag = "B-" + tag
    mid_tag = "I-" + tag
    end_tag = "E-" + tag
    o_tag = "O"
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags
