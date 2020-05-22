import os
import pickle
import time
import random
import torch
torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pointer_generator import GeneratorConfig as config
from pointer_generator import PointerGenerator


def collate_fn(insts, pad_id=1):
    # if seq_pad in class then all seqs with same length
    maxlen = max([len(x) for x in insts])
    seq = np.array([x + [pad_id] * (maxlen - len(x)) for x in insts])
    seq_lens = np.array([len(x) for x in insts])
    return torch.LongTensor(seq), torch.LongTensor(seq_lens)


def paired_collate_fn(insts):
    #src_insts, tgt_insts = list(zip(*insts))
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts = zip(*seq_pairs)
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)


class PhraseData(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def train_epoch(model, device, epoch, train_dataloader, validation_dataloader, optimizer, clip=5.):
    model.train()
    train_loss = 0
    t0 = time.time()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader, 1):
        enc_batch, enc_lens, dec_batch, dec_lens = batch
        optimizer.zero_grad()
        loss = model(enc_batch, enc_lens, dec_batch, dec_lens)
        # print(loss)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if step % 200 == 0:
            # print loss info every 20 Iterations
            dt = time.time() - t0
            log_str = f"Epoch: {epoch}, Iter: {step}, Time: {dt:.2f}, TrainLoss: {train_loss/step:.5f}"
            print(log_str)
            t0 = time.time()
    train_loss /= len(train_dataloader)

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloader, 1):
            enc_batch, enc_lens, dec_batch, dec_lens = batch
            loss = model(enc_batch, enc_lens, dec_batch, dec_lens)
            eval_loss += loss.item()
        eval_loss /= len(validation_dataloader)

    return model, optimizer, train_loss, eval_loss


def main(batch_size=16, multi_gpus=False, n_epochs=9):
    weight_path = './data/pg_emb_weight_v3.npy'
    save_path = './data/pg_model_v3.pt'
    with open('./data/e3_pg_process_data_v3.pkl', 'rb') as f:
        _process_data = pickle.load(f)

    _, train_src, train_tgt = _process_data['train']
    _, val_src, val_tgt = _process_data['val']

    # Create the DataLoader for our training set.
    train_loader = DataLoader(PhraseData(train_src, train_tgt),
                        batch_size=batch_size,
                        collate_fn=paired_collate_fn,
                        shuffle=True,
                        drop_last=True)

    # Create the DataLoader for our validation set.
    val_loader = DataLoader(PhraseData(val_src, val_tgt),
                        batch_size=batch_size,
                        collate_fn=paired_collate_fn,
                        shuffle=False,
                        drop_last=True)
    print(f"Train Data size : {len(train_loader)} * {batch_size}")
    print(f"Val Data size : {len(val_loader)} * {batch_size}")

    # Initializing model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    emb_weights = np.load(weight_path)
    model = PointerGenerator(device, weights=emb_weights)
    if multi_gpus:
        model = nn.DataParallel(model, device_ids=[0, 1], dim=0)
    # state_dict = torch.load(save_path)
    # model.load_state_dict(state_dict)
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    print('========= Begin Training ==========')
    clip = config.max_grad_norm
    best_eval_loss = float('inf')
    for epoch in range(1, 1+n_epochs):
        model, optimizer, train_loss, eval_loss = train_epoch(
            model, device, epoch, train_loader, val_loader, optimizer, clip=clip
        )

        print(f">> Epoch: {epoch}, TrainLoss: {train_loss:.5f}, EvalLoss: {eval_loss:.5f}\n")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), './models/pg_model_v31.pt')


if __name__ == '__main__':
    main(batch_size=64, multi_gpus=False, n_epochs=40)


# ===================== Draft model code =====================
# def init_lstm_wt(lstm):
#     for names in lstm._all_weights:
#         for name in names:
#             if name.startswith('weight_'):
#                 wt = getattr(lstm, name)
#                 wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
#             elif name.startswith('bias_'):
#                 # set forget bias to 1
#                 bias = getattr(lstm, name)
#                 n = bias.size(0)
#                 start, end = n // 4, n // 2
#                 bias.data.fill_(0.)
#                 bias.data[start:end].fill_(1.)

# def init_linear_wt(linear):
#     linear.weight.data.normal_(std=config.trunc_norm_init_std)
#     if linear.bias is not None:
#         linear.bias.data.normal_(std=config.trunc_norm_init_std)

# def init_wt_normal(wt):
#     wt.data.normal_(std=config.trunc_norm_init_std)

# def init_wt_unif(wt):
#     wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

# class Encoder(nn.Module):
#     def __init__(self, weights=None):
#         super(Encoder, self).__init__()
#         if weights is not None:
#             self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
#             init_wt_normal(self.embedding.weight)

#         self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
#         init_lstm_wt(self.lstm)

#         self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
#         self.dropout1 = nn.Dropout(p=config.dropout)
#         self.dropout2 = nn.Dropout(p=config.dropout)
#         self.layer_norm = nn.LayerNorm(2 * config.hidden_dim)

#     def forward(self, input_seq, input_lens):
#         embedded = self.embedding(input_seq)
#         embedded = self.dropout1(embedded)

#         packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
#         self.lstm.flatten_parameters()
#         output, hidden = self.lstm(packed)
#         encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # B x t_k x 2hs

#         encoder_feature = self.W_h(self.dropout2(encoder_outputs)) # B x t_k x 2hs
#         encoder_feature = self.layer_norm(encoder_feature)

#         return encoder_outputs, encoder_feature, hidden # hidden : tuple


# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention, self).__init__()
#         # attention
#         self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
#         self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
#         # self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

#     def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
#         # coverage_feature = self.W_c(coverage.unsqueeze(2)) # b x t_k x 1 - b x t_k x 2hs
#         # att_features = encoder_feature + coverage_feature
#         # scores = torch.bmm(att_features, s_t_hat.unsqueeze(2)) # b x t_k x 1
#         # scores = scores.squeeze(2)  # B x t_k

#         # attn_dist = F.softmax(scores + enc_padding_mask, dim=1)

#         # # attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
#         # c_t = torch.bmm(attn_dist.unsqueeze(1), encoder_outputs)  # B x 1 x t_k -> B x 1 x 2hs
#         # c_t = c_t.squeeze(1)  # B x 2hs

#         # coverage = coverage + attn_dist # B x t_k
#         # # output : B x 2hs, B x t_k, B x t_k
#         # return c_t, attn_dist, coverage

#         dec_fea = self.decode_proj(s_t_hat) # b x 2hs
#         # dec_fea_expanded = dec_fea.unsqueeze(1).expand_as(encoder_outputs).contiguous() # B x t_k x 2hs
#         coverage_feature = self.W_c(coverage.unsqueeze(2)) # b x t_k x 1 - b x t_k x 2hs

#         att_features = encoder_feature + coverage_feature # b x t_k x 2hs

#         scores = torch.bmm(att_features, dec_fea.unsqueeze(2))
#         scores = scores.squeeze(2)  # B x t_k

#         attn_dist = F.softmax(scores + enc_padding_mask, dim=1)

#         # attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
#         c_t = torch.bmm(attn_dist.unsqueeze(1), encoder_outputs)  # B x 1 x t_k - B x 1 x 2hs
#         c_t = c_t.squeeze(1)  # B x 2hs

#         coverage = coverage + attn_dist # B x t_k
#         # output : B x 2hs, B x t_k, B x t_k
#         return c_t, attn_dist, coverage


# class Decoder(nn.Module):
#     def __init__(self, weights=None):
#         super(Decoder, self).__init__()

#         # decoder
#         if weights is not None:
#             self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
#             init_wt_normal(self.embedding.weight)

#         self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

#         self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
#         init_lstm_wt(self.lstm)

#         self.attn = Attention()

#         self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

#         # p_vocab
#         self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
#         self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
#         init_linear_wt(self.out2)

#         self.dropout_context = nn.Dropout(p=config.dropout)
#         self.dropout1 = nn.Dropout(p=config.dropout)
#         # self.dropout2 = nn.Dropout(p=config.dropout)

#         # self.layer_norm = nn.LayerNorm(2 * config.hidden_dim)

#     def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
#                 c_t_1, enc_batch, coverage):
#         # c_t_1 : b x 2hs; extra_zeros: b x max_art_oovs
#         y_t_1_embd = self.embedding(y_t_1) # b x emb_dim

#         x = self.x_context( torch.cat((c_t_1, y_t_1_embd), 1) ) # b x (2hs+ed) - b x emb_dim
#         x = self.dropout_context(F.relu(x))
#         self.lstm.flatten_parameters()
#         lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1) # b x 1 x hs

#         h_decoder, c_decoder = s_t # both : 1 x b x hs
#         s_t_hat = torch.cat((h_decoder[-1], c_decoder[-1]), 1)  # b x 2hs

#         #  B x 2hs, B x t_k, B x t_k (c_t, attn_dist, coverage_next)
#         c_t, attn_dist, coverage_next = self.attn(s_t_hat, encoder_outputs, encoder_feature,
#                                                   enc_padding_mask, coverage)

#         # if self.training or step > 0:
#         coverage = coverage_next

#         # generation or copy
#         p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2hs + 2hs + emb_dim)
#         p_gen = self.p_gen_linear(p_gen_input) # b x 1
#         p_gen = torch.sigmoid(p_gen)

#         output = torch.cat((lstm_out.squeeze(1), c_t), 1) # B x 3hs
#         output = self.dropout1(F.relu(self.out1(output))) # B x hs

#         output = self.out2(output) # B x vocab_size
#         vocab_dist = F.softmax(output, dim=1)

#         vocab_dist_ = p_gen * vocab_dist
#         attn_dist_ = (1 - p_gen) * attn_dist

#         final_dist = vocab_dist_.scatter_add(1, enc_batch, attn_dist_)

#         return final_dist, s_t, c_t, attn_dist, p_gen, coverage

# class PointerGenerator(nn.Module):
#     def __init__(self, device, weights=None):
#         super(PointerGenerator, self).__init__()
#         self.encoder = Encoder(weights)
#         self.decoder = Decoder(weights)
#         self.device = device

#         # shared the embedding between encoder and decoder
#         self.decoder.embedding.weight = self.encoder.embedding.weight

#     def forward(self, enc_batch, enc_lens, dec_batch, dec_lens):
#         batch_size = enc_batch.size(0)
#         max_dec_len = dec_batch.size(1)

#         enc_batch = enc_batch.to(self.device)
#         dec_batch = dec_batch.to(self.device)
#         dec_lens = dec_lens.float().to(self.device)
#         # enc_padding_mask = (enc_batch != config.pad_id).float().to(self.device)
#         enc_padding_mask = (enc_batch == config.pad_id).float() * (-100000.0)
#         enc_padding_mask = enc_padding_mask.to(self.device)
#         dec_padding_mask = (dec_batch != config.pad_id).float().to(self.device)
#         c_t_1 = torch.zeros(batch_size, 2*config.hidden_dim).float().to(self.device)
#         coverage = torch.zeros_like(enc_batch).float().to(self.device)

#         encoder_outputs, encoder_feature, encoder_hidden = self.encoder(enc_batch, enc_lens)
#         h, c = encoder_hidden # h, c : 2 x b x hidden_dim
#         s_t_1 = (h[0].unsqueeze(0), c[0].unsqueeze(0)) # h, c : 1 x b x hidden_dim


#         step_losses = []
#         y_t_1 = torch.LongTensor([config.start_id] * batch_size).to(self.device)
#         # dec_outputs = torch.zeros(batch_size, max_tgt_len, config.vocab_size).to(self.device)
#         for di in range(max_dec_len):
#             final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = \
#                     self.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
#                                  enc_batch, coverage)

#             target = dec_batch[:, di]
#             # output:
#             teacher_force = random.random() < config.teacher_forcing_ratio
#             dec_out = F.log_softmax(final_dist, dim=1)
#             # dec_outputs[:, di, :] = dec_out
#             _, topi = dec_out.max(1)
#             y_t_1 = target if teacher_force else topi

#             # loss:
#             gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
#             step_loss = -torch.log(gold_probs + config.eps)

#             step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
#             step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
#             coverage = next_coverage

#             step_mask = dec_padding_mask[:, di]
#             step_loss = step_loss * step_mask
#             step_losses.append(step_loss)

#         sum_losses = torch.sum(torch.stack(step_losses, 1), 1) ## stack : [B x length] -> sum : [B]
#         batch_avg_loss = sum_losses/dec_lens
#         loss = torch.mean(batch_avg_loss)
#         return loss
