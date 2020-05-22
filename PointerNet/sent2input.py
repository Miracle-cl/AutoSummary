import re
import pandas as pd
from sqlalchemy import create_engine
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pymysql

from dataset_oc import OutcomeDatasets, paired_collate_fn
from oc_ptrnet import *

cat_cn = 'mysql+pymysql://'
cat_engine = create_engine(cat_cn)

class BeamState:
    def __init__(self, score, sents, hidden):
        self.score = score # probs of sent list
        self.sents = sents # word index list
        self.hidden = hidden

def read_case_name(own='extractor', version=3, short=False):
    lni_names = {}
    ## choose short_case_name or full_case_name
    if short:
        sql_name = ''
        names_df = pd.read_sql_query(sql_name, cat_engine)
        for row in names_df.itertuples():
            _, lni, short_case_name = row
            names = short_case_name.lower()
            names = [n.strip() for n in names.split(";")]
            if len(names) == 2:
                p_d = [(names[0],'plaintiff'), (names[1], 'defendant')]
            else:
                p_d = None
            if lni not in lni_names:
                lni_names[lni] = p_d
    else:
        sql_name = """cci"""
        names_df = pd.read_sql_query(sql_name, cat_engine)
        for row in names_df.itertuples():
            _, lni, full_case_name = row
            names = full_case_name.lower()
            split_tag = "vs." if "vs." in names else "v."
            names = [n.strip() for n in names.split(split_tag)]
            if len(names) == 2:
                p_d = [(names[0],'plaintiff'), (names[1], 'defendant')]
            else:
                p_d = None
            if lni not in lni_names:
                lni_names[lni] = p_d
    return lni_names


def replace_name(p_d, sent):
    # p_d : [('henry','plaintiff'), ('davis', 'defendant')]
    words = sent.split()
    for i, word in enumerate(words):
        if "'" in word:
            name = word.split("'")[0]
            if name in p_d[0][0]:
                words[i] = p_d[0][1] + "'s"
            elif name in p_d[1][0]:
                words[i] = p_d[1][1] + "'s"
    return " ".join(words)

def sub_abbreviation(sent):
    sent = re.sub(r"r&r |r & r ", r"report and recommendation ", sent)
    sent = re.sub(r"m&r |m & r ", r"memorandum and recommendation ", sent)
    return sent


def re_sub(sent):
    # grant, grants - granted
    if "remand" in sent and "to remand" not in sent:
        sent = re.sub(r"remand ", r"remanded ", sent)
    if "dismiss" in sent and "to dismiss" not in sent:
        sent = re.sub(r"dismiss ", r"dismissed ", sent)
    sent = re.sub(r"deny |denying |denies |deneid ", r"denied ", sent)
    sent = re.sub(r"grant |granting |grants ", r"granted ", sent)
    sent = re.sub(r"sustain |sustaining |sustains ", r"sustained ", sent)
    sent = re.sub(r"overrule |overruling |overrules ", r"overruled ", sent)
    sent = re.sub(r"affirm |affirming |affirms ", r"affirmed ", sent)
    sent = re.sub(r"accept |accepting |accepts ", r"accepted ", sent)
    sent = re.sub(r"adopt |adopting |adopts ", r"adopted ", sent)
    sent = re.sub(r"reverse |reversing |reverses ", r"reversed ", sent)
    sent = re.sub(r"reject |rejecting |rejects ", r"rejected ", sent)
    sent = re.sub(r"dismissing |dismisses ", r"dismissed ", sent)
    sent = re.sub(r"recommend |recommends |recommending ", r"recommended ", sent)
    return sent


def normilize_sent(s):
    # s: string
    s = s.lower()
    s = re.sub(r"([.,;])", r" \1", s)
    s = sub_abbreviation(s)
    s = re.sub(r"[^a-z.,;']+", r" ", s)
    s = re_sub(s)
    return s


def load_data_for_input(data_paths):
    # data_pkl, src_matrix, tgt_matrix = data_paths['data_pkl'], data_paths['src_matrix'], data_paths['tgt_matrix']
    data_pkl, emb_matrix_path = data_paths['data_pkl'], data_paths['emb_matrix']
    with open(data_pkl, 'rb') as pl:
        data_1 = pickle.load(pl)

    # xtrain, ytrain, yids_train = data_1['train']['x'], data_1['train']['y'], data_1['train']['yloc']
    # xtest, ytest, yids_test = data_1['test']['x'], data_1['test']['y'], data_1['test']['yloc']
    word2idx = data_1['word2idx']
    idx2word = data_1['idx2word']
    embedding_matrix = np.load(emb_matrix_path)

    return word2idx, idx2word, embedding_matrix  # , generate_loader


def generate(model, generate_loader, idx2word):
    # greedy algorithms for generate
    model.eval()
    batch_size = 1
    max_tgt_len = 15
    final_res = []
    with torch.no_grad():
        for j, batch in enumerate(generate_loader, 1):
            src_insts, src_lens, tgt_insts, tgt_lens, tgt_ids, _ = batch
            input_sents = ' '.join([idx2word[i.item()] for i in src_insts.view(-1)])
            # target_sents = ' '.join([idx2word[i.item()] for i in tgt_insts.view(-1)])

            src_insts = src_insts.to(model.device)
            # tgt_insts = tgt_insts.to(model.device)
            # tgt_ids = tgt_ids.to(model.device)
            enc_outs, last_hidden = model.encoder(src_insts, src_lens)
            mask_tensor = model.encoder.initial_mask(src_insts).to(model.device) # all zeros

            dec_input = torch.LongTensor([0] * batch_size).to(model.device)
            dec_words = []
            for step in range(max_tgt_len):
                att_weights, last_hidden = model.decoder(dec_input, last_hidden, enc_outs, mask_tensor)
                _, topi = att_weights.topk(1)
                dec_input_item = src_insts[ 0, topi.item() ].item()
                dec_words.append(dec_input_item)
                if dec_input_item == 10: # EOS_token == '.' == 10
                    break
                dec_input = torch.LongTensor([dec_input_item] * batch_size).to(model.device)

            output_sents = ' '.join([idx2word[w] for w in dec_words])
            final_res.append( [input_sents, output_sents] )
    return final_res


def generate_beam(model, generate_loader, idx2word, beam_size=2):
    # beam search for generate
    model.eval()
    batch_size = 1
    max_tgt_len = 15
    final_res = []
    with torch.no_grad():
        for j, batch in enumerate(generate_loader, 1):
            src_insts, src_lens, tgt_insts, tgt_lens, tgt_ids, _ = batch
            input_sents = ' '.join([idx2word[i.item()] for i in src_insts.view(-1)])
            # target_sents = ' '.join([idx2word[i.item()] for i in tgt_insts.view(-1)])

            src_insts = src_insts.to(model.device)
            # tgt_insts = tgt_insts.to(model.device)
            # tgt_ids = tgt_ids.to(model.device)
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
                        if bstate.sents[-1] == 10: # '.' == 10
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
            final_res.append( output_sent )
    return final_res


def main(data_paths, model_path, own='extractor', version=6, dot_id=10):
    ## read case names
    lni_names = read_case_name(own, version, short=False)

    # read data

    sql_ln = """mmmm"""
    ln_df = pd.read_sql_query(sql_ln, cat_engine)

    word2idx, idx2word, embedding_matrix = load_data_for_input(data_paths)
    # generate outcome
    allinputs = []
    all_oc_label_ids = []
    for row in ln_df.itertuples():
        _, idx, lni, sn, sents = row

        p_d = lni_names[lni] # for replacing names
        sents = normilize_sent(sents)

        ## replace name to plaintiff or defendant
        if p_d is not None:
            sents = replace_name(p_d, sents)

        src_ints = [word2idx[word] for word in sents.split() if word in word2idx]
        if src_ints[-1] != dot_id:
            src_ints.append(dot_id) # dot_id : '.'
        allinputs.append(src_ints)
        all_oc_label_ids.append(idx)

    # create any target for dataloader
    all_tgts = [[10] for _ in range(len(allinputs))] # useless
    all_tgts_ids = [[0] for _ in range(len(allinputs))] # useless
    generate_loader = torch.utils.data.DataLoader(
                            OutcomeDatasets(allinputs, all_tgts, all_tgts_ids),
                            num_workers = 1,
                            batch_size = 1,
                            collate_fn = paired_collate_fn,
                            shuffle = False,
                            drop_last = True)

    # model config and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerNet(300, 128, device, weights=embedding_matrix)

    model.load_state_dict(torch.load(model_path))
    model.to(model.device)

    # generate
    final_res1 = generate(model, generate_loader, idx2word)
    # beam_size = 2
    final_res2 = generate_beam(model, generate_loader, idx2word, beam_size = 2)
    final_res3 = generate_beam(model, generate_loader, idx2word, beam_size = 3)
    assert len(final_res1) == len(final_res2) and len(final_res1) == len(final_res3) and len(all_oc_label_ids) == len(final_res3)
    result2sql = []
    for i in range(len(all_oc_label_ids)):
        result2sql.append( (1, all_oc_label_ids[i], final_res1[i][0], '', '', final_res1[i][1],
                            final_res2[i], final_res3[i]) )
    merge_df = pd.DataFrame.from_records(result2sql, columns=['version', 'outcome_label_id', 'content', 'rewrite_sent', 'rewrite_oc', 'poc1', 'poc2', 'poc3'])
    # merge_df.to_sql('cc_ptr_result', con=cat_engine, index=False, if_exists='append')
    merge_df.to_csv('ptr_res_0320.csv', index=False)


if __name__ == '__main__':
    data_paths = {'data_pkl': 'data3.pkl',
                  'emb_matrix': 'EMatrix_3.npy'}
    model_path = 'oc_ptrnet_3.pt' # 1 is better
    main(data_paths, model_path)
