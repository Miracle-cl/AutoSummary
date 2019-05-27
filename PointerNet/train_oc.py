import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset_oc import OutcomeDatasets, paired_collate_fn
from oc_ptrnet import *


def train_epoch_pack(model, epoch, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    t0 = time.time()
    for i, batch in enumerate(train_loader, 1):
        src_insts, src_lens, tgt_insts, tgt_lens, tgt_ids, _ = batch
        src_insts = src_insts.to(device)
        tgt_insts = tgt_insts.to(device)
        tgt_ids = tgt_ids.to(device)
        optimizer.zero_grad()
        outputs = model(src_insts, src_lens, tgt_insts)
        loss = criterion(outputs.view(-1, outputs.size(2)), tgt_ids.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        train_loss += loss.item()
        if i % 15 == 0:
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.4f}".format \
                                (epoch, i, (time.time()-t0)/60., train_loss/i)
            print(log_str)
            t0 = time.time()

    train_loss = train_loss / len(train_loader)
    # print(train_loss)

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            src_insts, src_lens, tgt_insts, tgt_lens, tgt_ids, _ = batch
            src_insts = src_insts.to(device)
            tgt_insts = tgt_insts.to(device)
            tgt_ids = tgt_ids.to(device)
            outputs = model(src_insts, src_lens, tgt_insts)
            loss = criterion(outputs.view(-1, outputs.size(2)), tgt_ids.view(-1))
            eval_loss += loss.item()
        eval_loss = eval_loss / len(test_loader)
        # print(eval_loss)

    return train_loss, eval_loss

def main(paths, batchsize=256):
    with open(paths['data'], 'rb') as pl:
        data_1 = pickle.load(pl)

    xtrain, ytrain, yids_train = data_1['train']['x'], data_1['train']['y'], data_1['train']['yloc']
    xtest, ytest, yids_test = data_1['test']['x'], data_1['test']['y'], data_1['test']['yloc']
    word2idx = data_1['word2idx']
    idx2word = data_1['idx2word']
    embedding_matrix = np.load(paths['embed'])

    train_loader = torch.utils.data.DataLoader(
                        OutcomeDatasets(xtrain, ytrain, yids_train),
                        num_workers = 2,
                        batch_size = batchsize,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        OutcomeDatasets(xtest, ytest, yids_test),
                        num_workers = 2,
                        batch_size = 79,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointerNet(300, 128, device, weights=embedding_matrix)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss(ignore_index=929) # <PAD> - 929

    n_epochs = 100
    loss_dict = {'train': [], 'eval': []}
    best_eval_loss = float('inf')
    MODEL_SAVE_PATH = paths['save_model']
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        trainloss, evalloss = train_epoch_pack(model, epoch, train_loader, test_loader, criterion, optimizer, device)
        loss_dict['train'].append(trainloss)
        loss_dict['eval'].append(evalloss)
        used_time = time.time() - start
        print("Epoch : {} , Time : {:.2f} , TrainLoss : {:.4f} , EvalLoss : {:.4f}".format(epoch, used_time, trainloss, evalloss))
        if evalloss < best_eval_loss:
            best_eval_loss = evalloss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(paths['loss'], 'wb') as pl:
        pickle.dump(loss_dict, pl)


if __name__ == '__main__':
    paths = {'data': './data3.pkl', 'embed': 'EMatrix_3.npy',
             'save_model': 'oc_ptrnet_3.pt', 'loss': './loss3.pkl'}
    main(paths, batchsize=256)
