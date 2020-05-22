import pickle
import torch
import torch.nn.functional as F

from typing import Tuple, List, Dict

from pointer_generator import PointerGenerator, GeneratorConfig, source2ids


with open('', 'rb') as f:
    phrase2code = pickle.load(f)


class ModelPredict:
    def __init__(self):
        gen_model_path = './models/pg_model_v31.pt'

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        # generation
        self.seq2seq = self.load_gen_model(gen_model_path, self.device)
        self.seq2seq.to(self.device)

    def generate(self, sent: str):
        tup = self.prepare_gen_model_data(sent)
        wordids = self._generate_id(*tup)
        codes = self._generate_code(wordids)
        return codes

    def multi_generate(self, sent_list: List[str]):
        g_codes = []
        for sent in sent_list:
            _codes = self.generate(sent)
            g_codes.append(_codes)
        g_codes = self.rm_duplicate(g_codes)
        return g_codes

    @staticmethod
    def rm_duplicate(inp: List) -> List:
        # inp = [[120, 263], [167], [120],
        #        [230, 504], [167,487], [167]]
        inp.sort(key=lambda x: len(x))
        res = []
        while inp:
            tmp = inp.pop()
            res += [c for c in tmp if c not in res]
        return res

    @staticmethod
    def load_gen_model(pg_model_path, device):
        model = PointerGenerator(device, weights=None)
        state_dict = torch.load(pg_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model

    def prepare_gen_model_data(self, sent):
        enc_inp = source2ids(sent, GeneratorConfig.w2i) # tokenize

        enc_batch = torch.LongTensor([enc_inp]).to(self.device)
        enc_lens = torch.LongTensor([len(enc_inp)]).to(self.device)
        enc_padding_mask = (enc_batch == GeneratorConfig.pad_id).float() * (-100000.0)
        enc_padding_mask = enc_padding_mask.to(self.device)

        c_t_1 = torch.zeros(GeneratorConfig.batch_size, 2*GeneratorConfig.hidden_dim).float().to(self.device)
        coverage = torch.zeros_like(enc_batch).float().to(self.device)
        y_t_1 = torch.LongTensor([GeneratorConfig.start_id] * GeneratorConfig.batch_size).to(self.device)
        return enc_batch, enc_lens, enc_padding_mask, c_t_1, coverage, y_t_1

    def _generate_id(self, enc_batch, enc_lens, enc_padding_mask, c_t_1, coverage, y_t_1):
        self.seq2seq.eval()
        with torch.no_grad():
            encoder_outputs, encoder_feature, encoder_hidden = self.seq2seq.encoder(enc_batch, enc_lens)

            h, c = encoder_hidden
            s_t_1 = (h[0].unsqueeze(0), c[0].unsqueeze(0))

            pre_id = GeneratorConfig.start_id
            preds = []

            for _ in range(GeneratorConfig.max_dec_len):
                final_dist, s_t_1,  c_t_1, _, _, next_coverage = self.seq2seq.decoder(
                    y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, enc_batch, coverage
                )
                dec_out = F.log_softmax(final_dist, dim=1)
                coverage = next_coverage

                _, topis = dec_out.topk(GeneratorConfig.adj_top)
                flag = True
                for i in range(GeneratorConfig.adj_top):
                    topi = topis[:, i]
                    pid = topi.item()
                    if pid == GeneratorConfig.end_id:
                        break
                    if (pre_id, pid) in GeneratorConfig.adj:
                        flag = False # 'match'
                        y_t_1 = topi
                        pre_id = pid
                        preds.append(pid)
                        break
                if flag:
                    break
        preds = preds[:-1] if preds and preds[-1] == GeneratorConfig.end_id else preds
        # assert all((preds[i-1], preds[i]) in adj for i in range(1, len(preds))), preds
        return preds

    @staticmethod
    def _generate_code(preds):
        pred_str = ' '.join(GeneratorConfig.i2w[i] for i in preds)
        phrases = [p.strip() for p in pred_str.split(',')]
        codes = [phrase2code[p] for p in phrases if p in phrase2code]
        return phrases

