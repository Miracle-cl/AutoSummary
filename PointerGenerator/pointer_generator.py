import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


with open('./generation_model_w2i_v3.pkl', 'rb') as f:
    w2i = pickle.load(f)


with open('./_adj_set.pkl', 'rb') as f:
    # adjacency list
    adj = pickle.load(f)


class GeneratorConfig:
    # Hyperparameters
    unk_id = 0
    pad_id = 1
    start_id = 2
    end_id = 3
    hidden_dim = 256
    emb_dim = 300
    batch_size = 1
    teacher_forcing_ratio = 0.5
    rand_unif_init_mag = 0.02
    trunc_norm_init_std = 1e-4
    cov_loss_wt = 1.0
    eps = 1e-12
    max_grad_norm = 5.0
    dropout = 0.2
    max_dec_len = 32
    adj_top = 5

    w2i = w2i
    i2w = {i: w for w, i in w2i.items()}
    adj = adj
    vocab_size = len(w2i)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-GeneratorConfig.rand_unif_init_mag, GeneratorConfig.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=GeneratorConfig.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=GeneratorConfig.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=GeneratorConfig.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-GeneratorConfig.rand_unif_init_mag, GeneratorConfig.rand_unif_init_mag)


class Encoder(nn.Module):
    def __init__(self, weights=None):
        super(Encoder, self).__init__()
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        else:
            self.embedding = nn.Embedding(GeneratorConfig.vocab_size, GeneratorConfig.emb_dim)
            init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(GeneratorConfig.emb_dim, GeneratorConfig.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(GeneratorConfig.hidden_dim * 2, GeneratorConfig.hidden_dim * 2, bias=False)
        self.dropout1 = nn.Dropout(p=GeneratorConfig.dropout)
        self.dropout2 = nn.Dropout(p=GeneratorConfig.dropout)
        self.layer_norm = nn.LayerNorm(2 * GeneratorConfig.hidden_dim)

    def forward(self, input_seq, input_lens):
        embedded = self.embedding(input_seq)
        embedded = self.dropout1(embedded)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(packed)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # B x t_k x 2hs

        encoder_feature = self.W_h(self.dropout2(encoder_outputs)) # B x t_k x 2hs
        encoder_feature = self.layer_norm(encoder_feature)

        return encoder_outputs, encoder_feature, hidden # hidden : tuple


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        self.decode_proj = nn.Linear(GeneratorConfig.hidden_dim * 2, GeneratorConfig.hidden_dim * 2)
        self.W_c = nn.Linear(1, GeneratorConfig.hidden_dim * 2, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):

        dec_fea = self.decode_proj(s_t_hat)
        coverage_feature = self.W_c(coverage.unsqueeze(2))

        att_features = encoder_feature + coverage_feature

        scores = torch.bmm(att_features, dec_fea.unsqueeze(2))
        scores = scores.squeeze(2)

        attn_dist = F.softmax(scores + enc_padding_mask, dim=1)

        c_t = torch.bmm(attn_dist.unsqueeze(1), encoder_outputs)
        c_t = c_t.squeeze(1)

        coverage = coverage + attn_dist
        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, weights=None):
        super(Decoder, self).__init__()

        # decoder
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        else:
            self.embedding = nn.Embedding(GeneratorConfig.vocab_size, GeneratorConfig.emb_dim)
            init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(GeneratorConfig.hidden_dim * 2 + GeneratorConfig.emb_dim, GeneratorConfig.emb_dim)

        self.lstm = nn.LSTM(GeneratorConfig.emb_dim, GeneratorConfig.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.attn = Attention()

        self.p_gen_linear = nn.Linear(GeneratorConfig.hidden_dim * 4 + GeneratorConfig.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(GeneratorConfig.hidden_dim * 3, GeneratorConfig.hidden_dim)
        self.out2 = nn.Linear(GeneratorConfig.hidden_dim, GeneratorConfig.vocab_size)
        init_linear_wt(self.out2)

        self.dropout_context = nn.Dropout(p=GeneratorConfig.dropout)
        self.dropout1 = nn.Dropout(p=GeneratorConfig.dropout)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, enc_batch, coverage):
        y_t_1_embd = self.embedding(y_t_1)

        x = self.x_context( torch.cat((c_t_1, y_t_1_embd), 1) )
        x = self.dropout_context(F.relu(x))
        self.lstm.flatten_parameters()
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder[-1], c_decoder[-1]), 1)

        c_t, attn_dist, coverage_next = self.attn(s_t_hat, encoder_outputs, encoder_feature,
                                                  enc_padding_mask, coverage)

        coverage = coverage_next

        # generation or copy
        p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.squeeze(1), c_t), 1)
        output = self.dropout1(F.relu(self.out1(output)))

        output = self.out2(output)
        vocab_dist = F.softmax(output, dim=1)

        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        final_dist = vocab_dist_.scatter_add(1, enc_batch, attn_dist_)

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class PointerGenerator(nn.Module):
    def __init__(self, device, weights=None):
        super(PointerGenerator, self).__init__()
        self.encoder = Encoder(weights)
        self.decoder = Decoder(weights)
        self.device = device

        # shared the embedding between encoder and decoder
        self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, enc_batch, enc_lens, dec_batch, dec_lens):
        batch_size = enc_batch.size(0)
        max_dec_len = dec_batch.size(1)

        enc_batch = enc_batch.to(self.device)
        dec_batch = dec_batch.to(self.device)
        dec_lens = dec_lens.float().to(self.device)
        # enc_padding_mask = (enc_batch != config.pad_id).float().to(self.device)
        enc_padding_mask = (enc_batch == GeneratorConfig.pad_id).float() * (-100000.0)
        enc_padding_mask = enc_padding_mask.to(self.device)
        dec_padding_mask = (dec_batch != GeneratorConfig.pad_id).float().to(self.device)
        c_t_1 = torch.zeros(batch_size, 2*GeneratorConfig.hidden_dim).float().to(self.device)
        coverage = torch.zeros_like(enc_batch).float().to(self.device)

        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(enc_batch, enc_lens)
        h, c = encoder_hidden
        s_t_1 = (h[0].unsqueeze(0), c[0].unsqueeze(0))

        step_losses = []
        y_t_1 = torch.LongTensor([GeneratorConfig.start_id] * batch_size).to(self.device)

        for di in range(max_dec_len):
            final_dist, s_t_1,  c_t_1, attn_dist, _, next_coverage = \
                    self.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                 enc_batch, coverage)

            target = dec_batch[:, di]
            # output:
            teacher_force = random.random() < GeneratorConfig.teacher_forcing_ratio
            dec_out = F.log_softmax(final_dist, dim=1)
            _, topi = dec_out.max(1)
            y_t_1 = target if teacher_force else topi

            # loss:
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + GeneratorConfig.eps)

            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_loss = step_loss + GeneratorConfig.cov_loss_wt * step_coverage_loss
            coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens
        loss = torch.mean(batch_avg_loss)
        return loss


def source2ids(text, w2i, unk_id=0):
    enc_inp = [w2i.get(w, unk_id) for w in text.split()]
    return enc_inp