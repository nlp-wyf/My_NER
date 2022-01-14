def get_accuracy(path):
    with open(path, "r", encoding='utf-8') as f:
        sents = [line.strip() for line in f.readlines() if line.strip()]

    total = len(sents)

    count = 0
    for sent in sents:
        words = sent.split()
        if words[-1] == words[-2]:
            count += 1

    print("Accuracy: %.4f" % (count / total))


if __name__ == '__main__':
    path = "results/test_result.txt"
    get_accuracy(path)
