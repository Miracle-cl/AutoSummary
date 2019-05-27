import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine

from dataset_oc import OutcomeDatasets, paired_collate_fn
from oc_ptrnet import *

class BeamState:
    def __init__(self, score, sents, hidden):
        self.score = score # probs of sent list
        self.sents = sents # word index list
        self.hidden = hidden

def load_data(paths):
    # data_pkl, src_matrix, tgt_matrix = data_paths['data_pkl'], data_paths['src_matrix'], data_paths['tgt_matrix']
    data_pkl, emb_matrix_path = paths['data'], paths['embed']
    with open(data_pkl, 'rb') as pl:
        data_1 = pickle.load(pl)

    xtrain, ytrain, yids_train = data_1['train']['x'], data_1['train']['y'], data_1['train']['yloc']
    xtest, ytest, yids_test = data_1['test']['x'], data_1['test']['y'], data_1['test']['yloc']
    word2idx = data_1['word2idx']
    idx2word = data_1['idx2word']
    embedding_matrix = np.load(emb_matrix_path)

    generate_loader = torch.utils.data.DataLoader(
                            OutcomeDatasets(xtest, ytest, yids_test),
                            num_workers = 1,
                            batch_size = 1,
                            collate_fn = paired_collate_fn,
                            shuffle = True,
                            drop_last = True)

    return word2idx, idx2word, embedding_matrix, generate_loader


def generate(model, generate_loader, idx2word, end_id=930):
    # greedy algorithms for generate
    model.eval()
    batch_size = 1
    max_tgt_len = 15
    final_res = {}
    with torch.no_grad():
        for j, batch in enumerate(generate_loader, 1):
            src_insts, src_lens, tgt_insts, tgt_lens, tgt_ids, _ = batch
            input_sents = ' '.join([idx2word[i.item()] for i in src_insts.view(-1)])
            target_sents = ' '.join([idx2word[i.item()] for i in tgt_insts.view(-1)])
            # print('> ', input_sents)
            # print('= ', target_sents)

            src_insts = src_insts.to(model.device)
            tgt_insts = tgt_insts.to(model.device)
            tgt_ids = tgt_ids.to(model.device)
            enc_outs, last_hidden = model.encoder(src_insts, src_lens)
            mask_tensor = model.encoder.initial_mask(src_insts).to(model.device) # all zeros

            dec_input = torch.LongTensor([0] * batch_size).to(model.device)
            dec_words = []
            for step in range(max_tgt_len):
                att_weights, last_hidden = model.decoder(dec_input, last_hidden, enc_outs, mask_tensor)
                _, topi = att_weights.topk(1)
                dec_input_item = src_insts[ 0, topi.item() ].item()
                dec_words.append(dec_input_item)
                if dec_input_item == end_id: # EOS_token == end_id
                    break
                dec_input = torch.LongTensor([dec_input_item] * batch_size).to(model.device)

            output_sents = ' '.join([idx2word[w] for w in dec_words])
            # print('< ', output_sents)
            # print()
            final_res[input_sents] = [target_sents, output_sents]
    return final_res


def generate_beam(model, generate_loader, idx2word, beam_size=2, end_id=930):
    # beam search for generate
    model.eval()
    batch_size = 1
    max_tgt_len = 15
    final_res = {}
    with torch.no_grad():
        for j, batch in enumerate(generate_loader, 1):
            src_insts, src_lens, tgt_insts, tgt_lens, tgt_ids, _ = batch
            input_sents = ' '.join([idx2word[i.item()] for i in src_insts.view(-1)])
            target_sents = ' '.join([idx2word[i.item()] for i in tgt_insts.view(-1)])
            # print('> ', input_sents)
            # print('= ', target_sents)

            src_insts = src_insts.to(model.device)
            tgt_insts = tgt_insts.to(model.device)
            tgt_ids = tgt_ids.to(model.device)
            enc_outs, last_hidden = model.encoder(src_insts, src_lens)
            mask_tensor = model.encoder.initial_mask(src_insts).to(model.device)

            dec_input = torch.LongTensor([0] * batch_size).to(model.device)
            res = []
            for step in range(max_tgt_len):
                flag = 0
                if step == 0:
                    att_weights, last_hidden = model.decoder(dec_input, last_hidden, enc_outs, mask_tensor)
                    topv, topi = att_weights.topk(beam_size)
                    for i in range(beam_size):
                        dec_input_item = src_insts[ 0, topi[0, i].item() ].item()
                        res.append( BeamState(topv[0, i].item(), [ dec_input_item ], last_hidden) )
                else:
                    prev_states = res[:beam_size]
                    next_states = []
                    for bstate in prev_states:
                        if bstate.sents[-1] == end_id: # EOS_token
                            next_states.append(bstate)
                            flag += 1
                            continue

                        dec_input = torch.LongTensor([bstate.sents[-1]] * batch_size).to(model.device)
                        att_weights, last_hidden = model.decoder(dec_input, bstate.hidden, enc_outs, mask_tensor)
                        topv, topi = att_weights.topk(beam_size)
                        for i in range(beam_size):
                            new_score = (bstate.score * len(bstate.sents) + topv[0, i].item()) / (len(bstate.sents) + 1) # log_softmax
                            # topi[0, i].item() : a pointer point location in input
                            dec_input_item = src_insts[ 0, topi[0, i].item() ].item()
                            new_sents = bstate.sents + [dec_input_item]
                            next_states.append( BeamState(new_score, new_sents, last_hidden) )
                    res = sorted(next_states, key=lambda x: x.score, reverse=True)
                    # print(j, step,  [x.score for x in res])
                if flag == beam_size:
                    break
            output_sent = ' '.join([idx2word[i] for i in res[0].sents])
            # print('< ', output_sent)
            # print()
            final_res[input_sents] = output_sent
    return final_res


def main(paths, version=2):
    word2idx, idx2word, embedding_matrix, generate_loader = load_data(paths)

    # model config and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerNet(300, 128, device, weights=embedding_matrix)

    model.load_state_dict(torch.load(paths['save_model']))
    model.to(model.device)

    # generate
    final_res1 = generate(model, generate_loader, idx2word)
    # beam_size = 2
    final_res2 = generate_beam(model, generate_loader, idx2word, beam_size = 2)
    final_res3 = generate_beam(model, generate_loader, idx2word, beam_size = 3)
    merge_res = []
    for k, v in final_res1.items():
        # k : input_sents
        # v[0] : target_sents - rewrite oc
        # v[1] : output_sents - poc1
        if k in final_res2 and k in final_res3:
            merge_res.append( (version, '', k, '', v[0], v[1], final_res2[k], final_res3[k]) )

    cat_cn = 'mysql+pymysql://'
    cat_engine = create_engine(cat_cn)
    merge_df = pd.DataFrame.from_records(merge_res, columns=['version', 'outcome_label_id', 'content',
                            'rewrite_sent', 'rewrite_oc', 'poc1', 'poc2', 'poc3'])
    merge_df.to_sql('cc_ptr_result', con=cat_engine, index=False, if_exists='append')


if __name__ == "__main__":
    data_paths = {'data': 'data2.pkl',
                  'embed': 'EMatrix_2.npy',
                  'save_model': 'oc_ptrnet_2.pt'}

    main(data_paths)
