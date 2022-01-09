import os
import json
from sklearn.model_selection import train_test_split


def Process_File(file_name, path, enc):
    with open(file_name, 'r', encoding=enc) as f:
        i = 0
        while True:
            txt = f.readline()
            if not txt:
                break  # end loop
            i += 1
            j = json.loads(txt)
            orig = j['originalText']  # original text
            entities = j['entities']  # entity part
            pathO = path + str(i) + '-original.txt'
            pathE = path + str(i) + '.txt'

            with open(pathO, 'w', encoding='utf-8') as o1:  # write the original text
                o1.write(orig)
                o1.flush()

            with open(pathE, 'w', encoding='utf-8') as o2:  # wirte entity file
                for e in entities:
                    start = e['start_pos']  # extract start position
                    end = e['end_pos']  # extract end position
                    name = orig[start:end]  # entity content
                    ty = e['label_type']  # entity label type
                    label = '{0}\t{1}\t{2}\t{3}\n'.format(name, start, end, ty)
                    o2.write(label)
                    o2.flush()


def sentence2BIOlabel(sentence, label_from_file):
    """ BIO Tagging """
    sentence_label = ['O'] * len(sentence)
    if label_from_file == '':
        return sentence_label

    for line in label_from_file.split('\n'):

        entity_info = line.strip().split('\t')
        start_index = int(entity_info[1])
        end_index = int(entity_info[2])
        entity_label = label_dict[entity_info[3]]
        # Frist entity: B-xx
        sentence_label[start_index] = 'B-' + entity_label
        # Other: I-xx
        for i in range(start_index + 1, end_index):
            sentence_label[i] = 'I-' + entity_label
    return sentence_label


def loadRawData(fileName):
    """ Loading raw data and tagging """
    sentence_list = []
    label_list = []

    for file_name in os.listdir(fileName):

        if '.DS_Store' == file_name:
            continue

        if 'original' in file_name:
            org_file = fileName + file_name
            lab_file = fileName + file_name.replace('-original', '')

            with open(org_file, encoding='utf-8') as f:
                content = f.read().strip()

            with open(lab_file, encoding='utf-8') as f:
                content_label = f.read().strip()

            sentence_label = sentence2BIOlabel(content, content_label)
            sentence_list.append(content)
            label_list.append(sentence_label)

    return sentence_list, label_list


def Save_data(filename, texts, tags):
    """ Processing to files in neeed format """
    with open(filename, 'w', encoding='utf-8') as f:
        for sent, tag in zip(texts, tags):
            size = len(sent)
            for i in range(size):
                f.write(sent[i])
                f.write('\t')
                f.write(tag[i])
                f.write('\n')


if __name__ == '__main__':
    FILE1 = './raw_data/subtask1_training_part1.txt'
    FILE2 = './raw_data/subtask1_training_part2.txt'
    FILE3 = './raw_data/subtask1_test_set_with_answer.json'

    PATH1 = './data_files/train/data1-'
    PATH2 = './data_files/train/data2-'
    PATH3 = './data_files/test/data-test-'

    Process_File(FILE1, PATH1, 'utf-8-sig')
    Process_File(FILE2, PATH2, 'utf-8-sig')
    Process_File(FILE3, PATH3, 'utf-8')

    # 6种实体类型
    label_dict = {'药物': 'DRUG',
                  '解剖部位': 'BODY',
                  '疾病和诊断': 'DISEASES',
                  '影像检查': 'EXAMINATIONS',
                  '实验室检验': 'TEST',
                  '手术': 'TREATMENT'}

    TRAIN = './dataset/train_dataset.txt'
    VALID = './dataset/val_dataset.txt'
    TEST = './dataset/test_dataset.txt'

    # Training data
    sentence_list, label_list = loadRawData('./data_files/train/')
    # Test data
    sentence_list_test, label_list_test = loadRawData('./data_files/test/')

    # Split dataset
    words = [list(sent) for sent in sentence_list]
    t_words = [list(sent) for sent in sentence_list_test]
    tags = label_list
    t_tags = label_list_test
    train_texts, val_texts, train_tags, val_tags = train_test_split(words, tags, test_size=.2)
    test_texts, test_tags = t_words, t_tags

    # Obtain training, validating and testing files
    Save_data(TRAIN, train_texts, train_tags)
    Save_data(VALID, val_texts, val_tags)
    Save_data(TEST, test_texts, test_tags)
