import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, weights=None, n_layers=2, dropout=0.5):
        super(Encoder, self).__init__()

        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        else:
            input_size = 1000
            self.embedding = nn.Embedding(input_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)

    def forward(self, input_seqs, input_lens):
        embeded = self.embedding(input_seqs)
        total_len = embeded.size(1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_lens, batch_first=True)
        # self.lstm.flatten_parameters()
        enc_outs, enc_hidden = self.lstm(packed)
        enc_outs, out_lens = torch.nn.utils.rnn.pad_packed_sequence(enc_outs, batch_first=True, total_length=total_len)

        last_hidden = ( torch.cat((enc_hidden[0][-2], enc_hidden[0][-1]), dim=-1).unsqueeze(0),
                        torch.cat((enc_hidden[1][-2], enc_hidden[1][-1]), dim=-1).unsqueeze(0) )
        # enc_outs : bs x seqlen x 2hs; last_hidden : (1 x bs x 2hs, 1 x bs x 2hs)
        return enc_outs, last_hidden

    def initial_mask(self, input_seqs):
        mask_tensor = (input_seqs == 0).float() * (-10000)
        return mask_tensor


class Attn(nn.Module):
    def __init__(self, hs):
        super(Attn, self).__init__()
        self.general_attn = nn.Linear(hs, hs)

    def forward(self, dec_out, enc_outs, mask_tensor):
        # dec_out : bs x 1 x 2hs
        # enc_outs : bs x seqlen x 2hs
        energy = self.general_attn(enc_outs).permute(0, 2, 1) # bs x 2hs x seqlen
        attn_energy = torch.bmm(dec_out, energy).squeeze(1) #  bs x seqlen
        attn_energy += mask_tensor # make pad = 0 to -10000
        attn_weight = F.log_softmax(attn_energy, dim=1)
        return attn_weight


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, weights=None, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()

        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        else:
            input_size = 1000
            self.embedding = nn.Embedding(input_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, 2 * hidden_size, n_layers, batch_first=True)
        self.attn = Attn(2 * hidden_size)


    def forward(self, input_seq, last_hidden, enc_outs, mask_tensor):
        embeded = self.embedding_dropout(self.embedding(input_seq)).unsqueeze(1) # bs x 1 x ed
        # self.lstm.flatten_parameters()
        dec_out, dec_hidden = self.lstm(embeded, last_hidden) # bs x 1 x 2hs
        att_weights = self.attn(dec_out, enc_outs, mask_tensor) #  bs x seqlen
        return att_weights, dec_hidden


class PointerNet(nn.Module):
    def __init__(self, embed_dim, hidden_size, device, weights=None):
        super(PointerNet, self).__init__()
        self.encoder = Encoder(embed_dim, hidden_size, weights=weights)
        self.decoder = Decoder(embed_dim, hidden_size, weights=weights)
        self.device = device

    def forward(self, input_seqs, input_lens, target_seqs, teacher_forcing_ratio=0.5):
        # target_ids is the index_id of target_seqs in input_seqs
        # target_ids is used for calculating CrossEntropyLoss
        # target_seqs used as decoder input with embedding
        batch_size, input_seq_len = input_seqs.size()
        max_len = target_seqs.size(1)
        enc_outs, last_hidden = self.encoder(input_seqs, input_lens)
        mask_tensor = self.encoder.initial_mask(input_seqs).to(self.device)

        dec_input = torch.LongTensor([0] * batch_size).to(self.device) # <sos> - 0
        # first max_len is the ith dec_out, second input_seq_len is output_size
        dec_outputs = torch.zeros(batch_size, max_len, input_seq_len).to(self.device)
        for i in range(max_len):
            att_weights, last_hidden = self.decoder(dec_input, last_hidden, enc_outs, mask_tensor)
            dec_outputs[:, i, :] = att_weights
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                dec_input = target_seqs[:, i]
            else:
                _, topi = att_weights.max(1)
                dec_input = input_seqs.gather(1, topi.view(-1, 1)).squeeze(1)
        return dec_outputs
