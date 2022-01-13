import torch
from torch.utils import data
import os
from tqdm import tqdm
import warnings
import argparse
import numpy as np
from sklearn import metrics
from bert_crf import BERT_CRF
from bert_bilstm_crf import Bert_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loader import NerDataset, PadBatch, VOCAB, tag2idx, idx2tag

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(tqdm(iterator)):
        step += 1

        tokens, labels, mask = batch
        tokens = tokens.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        loss = model(tokens, labels, mask)
        losses += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses / step))


def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            step += 1

            tokens, labels, mask = batch
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            y_hat = model(tokens, labels, mask, is_test=True)
            loss = model(tokens, labels, mask)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (mask == 1)
            y_orig = torch.masked_select(labels, mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean() * 100

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses / step, acc))
    return model, losses / step, acc


def test(model, iterator, device):
    model.load_state_dict(torch.load('./ckpt/bert_bilstm_crf.pth'))
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            tokens, labels, mask = batch
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            y_hat = model(tokens, labels, mask, is_test=True)
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (mask == 1).cpu()
            y_orig = torch.masked_select(labels, mask)
            Y.append(y_orig)

    Y = torch.cat(Y, dim=0).cpu().numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]

    return y_true, y_pred


def run():
    labels = ['B-BODY',
              'B-DISEASES',
              'B-DRUG',
              'B-EXAMINATIONS',
              'B-TEST',
              'B-TREATMENT',
              'I-BODY',
              'I-DISEASES',
              'I-DRUG',
              'I-EXAMINATIONS',
              'I-TEST',
              'I-TREATMENT']

    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--train_set", type=str, default="./CCKS_2019_Task1/dataset/train_dataset.txt")
    parser.add_argument("--valid_set", type=str, default="./CCKS_2019_Task1/dataset/val_dataset.txt")
    parser.add_argument("--test_set", type=str, default="./CCKS_2019_Task1/dataset/test_dataset.txt")

    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = BERT_CRF(tag2idx).to(device)
    model = Bert_BiLSTM_CRF(tag2idx).to(device)

    print('Initial model Done.')
    train_dataset = NerDataset(ner.train_set)
    eval_dataset = NerDataset(ner.valid_set)
    test_dataset = NerDataset(ner.test_set)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=ner.batch_size // 2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=ner.batch_size // 2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset)
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch \
        if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch

    warm_up_ratio = 0.1  # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    print('Start Train...')
    for epoch in range(1, ner.n_epochs + 1):
        train(epoch, model, train_iter, optimizer, scheduler, device)
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device)

        if loss < _best_val_loss and acc > _best_val_acc:
            _best_val_loss = loss
            _best_val_acc = acc
            torch.save(candidate_model.state_dict(), "./ckpt/bert_bilstm_crf.pth")

        print("=============================================")

    print('Eval Metrics...')
    y_test, y_pred = test(model, test_iter, device)
    print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))


if __name__ == "__main__":
    run()
