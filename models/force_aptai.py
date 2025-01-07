import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from w2v2_pr import(
    Wav2Vec2_PR
)
from modules import(
    CrossAttention,
    ForwardSumLoss,
    PositionalEncoding,
    LowPassFilterLayer,
    RNN,
)


class Force_APTAI(nn.Module):
    def __init__(self, pr_model_path, device, vocab):
        super().__init__()
        assert os.path.exists(pr_model_path)
        
        self.vocab = vocab
        self.device = device
        self.i = 0 # for debugging
        
        self.hidden_drop = 0.2
        self.rnn_drop = 0.1
        self.max_phn_seq_len = 60
        self.frame_hidden_dim = 128
        self.phn_hidden_dim = 128
        self.att_hidden_dim = 128
        self.rnn_in_dim = 2 * self.att_hidden_dim
        
        # Cross Attention
        self.xatt = CrossAttention(
            self.frame_hidden_dim,
            self.phn_hidden_dim,
            self.att_hidden_dim,
        )
        self.align_loss = ForwardSumLoss()
        # frame linear
        self.frame_lin = nn.Linear(1024, self.frame_hidden_dim)
        self.frame_drop = nn.Dropout(self.hidden_drop)
        # Phoneme
        self.phn_emb_layer = nn.Embedding(
            len(self.vocab),
            self.phn_hidden_dim,
            padding_idx=0
        )
        self.pe_phn = PositionalEncoding(
            self.phn_hidden_dim,
            max_len=60,
            dropout=self.hidden_drop
        )
        # TVs RNN
        self.rnn = RNN(self.rnn_in_dim, 9, self.rnn_drop)
        self.tv_lowpass = LowPassFilterLayer(self.device, 10, 49, 9)
        # wav2vec2 phoneme recognizer
        self.pr_model_path = pr_model_path
        pr_ckpt_path = os.path.join(pr_model_path, 'best-model-ckpt')
        self.w2v2_pr_cfg = pickle.load(
            open(os.path.join(pr_ckpt_path, 'model_cfg.pkl'), 'rb')
        )
        self.w2v2_pr = Wav2Vec2_PR(
            self.w2v2_pr_cfg['pretrain_cfg'],
            self.w2v2_pr_cfg['cache_dir'],
            self.w2v2_pr_cfg['huggingface_model_id'],
            vocab
        ).to(self.device)
        self.w2v2_pr.load_state_dict(torch.load(
            os.path.join(pr_ckpt_path, 'pytorch_model.bin'),
            map_location=torch.device(self.device)
        ))
        # freeze entire w2v2_pr (True here, to unfreeze entire thing)
        for param in self.w2v2_pr.parameters():
            param.requires_grad = False

    def forward(
        self, epoch,
        audio_inputs, audio_lengths,
        phoneme_labels, phn_frames_49hz,
        LA, LP, JA,
        TTCL, TTCD,
        TMCL, TMCD,
        TBCL, TBCD,
        ):
        # TV's 
        tv_targets = torch.stack([LA, LP, JA,
                              TTCL, TTCD,
                              TMCL, TMCD,
                              TBCL, TBCD], dim=-1).float()
        tv_pad_mask = (tv_targets != -100.0)
        
        # w2v2 phoneme recognizer "ac" == acoustic
        w2v2_pr_out = self.w2v2_pr.get_embeddings(
            audio_inputs,
            audio_lengths,
        )
        ac_frame_embs = w2v2_pr_out['last_transf_hidden']
        phn_pred_list = w2v2_pr_out['phn_pred_seq_idx']

        frame_seq_lens = w2v2_pr_out['frame_seq_lens'].tolist()
        phn_seq_lens = []
        for lst in phn_pred_list:
            phn_seq_lens.append(len(lst))
        
        phn_pred_seq = []
        for lst in phn_pred_list:
            assert len(lst) < self.max_phn_seq_len, 'Need longer max phoneme sequence length.'
            padded = np.pad(lst, (0, self.max_phn_seq_len - len(lst)), mode='constant')
            phn_pred_seq.append(padded)
        phn_pred_seq = torch.tensor(phn_pred_seq, dtype=torch.int32, device=self.device)
        phn_pred_mask = (phn_pred_seq != 0).to(torch.int)

        # ctc predicted phonemes embedding
        phn_embs = self.phn_emb_layer(phn_pred_seq)
        phn_embs = self.pe_phn(phn_embs.permute(1,0,2)).permute(1,0,2)
        
        # frame linear
        frame_hidden_emb = self.frame_lin(ac_frame_embs.permute(0,2,1))
        frame_hidden_emb = self.frame_drop(frame_hidden_emb)

        # cross attention matrix
        att_out, energy  = self.xatt(frame_hidden_emb, phn_embs, phn_pred_mask)
        # since FS loss needs log_softmax
        att_mask = (1 - phn_pred_mask) * - 1000.0
        att_mask = att_mask.unsqueeze(1).repeat(1, energy.size(1), 1)
        att = torch.log_softmax(energy + att_mask, dim=-1)
        
        # auto-regressive part
        rnn_out = self.rnn(att_out, frame_seq_lens)
        tvs_out = self.tv_lowpass(rnn_out[0])

        # Loss
        tv_loss = F.mse_loss(
            input=tvs_out[tv_pad_mask],
            target=tv_targets[tv_pad_mask],
            reduction='mean',
        )
        align_loss = self.align_loss(att.unsqueeze(1), phn_seq_lens, frame_seq_lens)

        a = 0.4
        loss = a * tv_loss + (1 - a) * align_loss

        # predicted frame wise phonemes
        align_out = torch.max(att, axis=2)[1]
        # this is the same as above
        # align_out = torch.softmax(energy, dim=-1)
        # align_preds = torch.argmax(align_out, dim=-1)
        pred_frame_align = []
        for batch_idx, align in enumerate(align_out):
            pred_frame_align.append(align[:frame_seq_lens[batch_idx]].detach().cpu().numpy())
        pred_frame_phns = []
        for batch_idx, align in enumerate(pred_frame_align):
            tmp = []
            for frame in align:
                pred_phn = int(phn_pred_seq[batch_idx][frame].detach().cpu().numpy())
                tmp.append(pred_phn)
            pred_frame_phns.append(tmp)

        # debugging
        # self.i += 1
        # if self.i % 200 == 0:
        #     test = att[0][0:frame_seq_lens[0],0:phn_seq_lens[0]]
        #     plt.imshow(test.detach().cpu().numpy().T)
        #     plt.savefig(f'../analysis/alignment/alignment_attention{self.i}.png')
        #     plt.close()
            
        return {
            'loss': loss,
            'tv_loss': tv_loss,
            'align_loss': align_loss,
            'tvs_pred': tvs_out,
            'pred_frame_phns': pred_frame_phns,
            'pred_ctc_phn_seq': phn_pred_list,
        }

    def get_config(self):
        return {
            'pr_model_path': self.pr_model_path,
            'w2v2_pr_cfg': self.w2v2_pr_cfg,
            'device': self.device,
            'vocab': self.vocab,
        }
    
    def get_alignment(self, wav):
        self.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        with torch.no_grad():
            if type(wav) is torch.Tensor:
                wav = wav[0]
            wav_input = torch.unsqueeze(torch.Tensor(wav), dim=0).to(torch.device(device))
            wav_len = torch.unsqueeze(torch.LongTensor([len(wav)]), dim=0).to(torch.device(device))
            
        w2v2_pr_out = self.w2v2_pr.get_embeddings(
            wav_input,
            wav_len,
        )
        ac_frame_embs = w2v2_pr_out['last_transf_hidden']
        phn_pred_list = w2v2_pr_out['phn_pred_seq_idx']

        frame_seq_lens = w2v2_pr_out['frame_seq_lens'].tolist()
        phn_seq_lens = []
        for lst in phn_pred_list:
            phn_seq_lens.append(len(lst))
        
        phn_pred_seq = []
        for lst in phn_pred_list:
            assert len(lst) < self.max_phn_seq_len, 'Need longer max phoneme sequence length.'
            padded = np.pad(lst, (0, self.max_phn_seq_len - len(lst)), mode='constant')
            phn_pred_seq.append(padded)
        phn_pred_seq = torch.tensor(phn_pred_seq, dtype=torch.int32, device=self.device)
        phn_pred_mask = (phn_pred_seq != 0).to(torch.int)

        # ctc predicted phonemes embedding
        phn_embs = self.phn_emb_layer(phn_pred_seq)
        phn_embs = self.pe_phn(phn_embs.permute(1,0,2)).permute(1,0,2)
        
        # frame linear
        frame_hidden_emb = self.frame_lin(ac_frame_embs.permute(0,2,1))
        frame_hidden_emb = self.frame_drop(frame_hidden_emb)

        # cross attention matrix
        att_out, energy  = self.xatt(frame_hidden_emb, phn_embs, phn_pred_mask)
        att_mask = (1 - phn_pred_mask) * - 1000.0
        att_mask = att_mask.unsqueeze(1).repeat(1, energy.size(1), 1)
        att = torch.log_softmax(energy + att_mask, dim=-1).squeeze(dim=0)

        align_result = att[0:frame_seq_lens[0][0], 0:phn_seq_lens[0]]
        align_result = align_result.permute(1,0)

        return {
            'alignment': align_result.detach().cpu().numpy()
        }

    def get_faptai_output(self, wav):
        self.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        with torch.no_grad():
            if type(wav) is torch.Tensor:
                wav = wav[0]
            wav_input = torch.unsqueeze(torch.Tensor(wav), dim=0).to(torch.device(device))
            wav_len = torch.unsqueeze(torch.LongTensor([len(wav)]), dim=0).to(torch.device(device))
            
        w2v2_pr_out = self.w2v2_pr.get_embeddings(
            wav_input,
            wav_len,
        )
        ac_frame_embs = w2v2_pr_out['last_transf_hidden']
        phn_pred_list = w2v2_pr_out['phn_pred_seq_idx']

        frame_seq_lens = w2v2_pr_out['frame_seq_lens'].tolist()
        phn_seq_lens = []
        for lst in phn_pred_list:
            phn_seq_lens.append(len(lst))
        
        phn_pred_seq = []
        for lst in phn_pred_list:
            assert len(lst) < self.max_phn_seq_len, 'Need longer max phoneme sequence length.'
            padded = np.pad(lst, (0, self.max_phn_seq_len - len(lst)), mode='constant')
            phn_pred_seq.append(padded)
        phn_pred_seq = torch.tensor(phn_pred_seq, dtype=torch.int32, device=self.device)
        phn_pred_mask = (phn_pred_seq != 0).to(torch.int)

        # ctc predicted phonemes embedding
        phn_embs = self.phn_emb_layer(phn_pred_seq)
        phn_embs = self.pe_phn(phn_embs.permute(1,0,2)).permute(1,0,2)
        
        # frame linear
        frame_hidden_emb = self.frame_lin(ac_frame_embs.permute(0,2,1))
        frame_hidden_emb = self.frame_drop(frame_hidden_emb)

        # cross attention matrix
        att_out, energy  = self.xatt(frame_hidden_emb, phn_embs, phn_pred_mask)
        att_mask = (1 - phn_pred_mask) * - 1000.0
        att_mask = att_mask.unsqueeze(1).repeat(1, energy.size(1), 1)
        att = torch.log_softmax(energy + att_mask, dim=-1).squeeze(dim=0)
        
        # auto-regressive part
        rnn_out = self.rnn(att_out, frame_seq_lens)
        tvs_out = self.tv_lowpass(rnn_out[0])

        # TVS
        tvs_out = tvs_out.squeeze(dim=0).detach().cpu().numpy()
        tvs_pred_dict = {
            'LA': [],
            'LP': [],
            'JA': [],
            'TTCL': [],
            'TTCD': [],
            'TMCL': [],
            'TMCD': [],
            'TBCL': [],
            'TBCD': [],
        }
        for tv in tvs_out:
            tvs_pred_dict['LA'].append(tv[0])
            tvs_pred_dict['LP'].append(tv[1])
            tvs_pred_dict['JA'].append(tv[2])
            tvs_pred_dict['TTCL'].append(tv[3])
            tvs_pred_dict['TTCD'].append(tv[4])
            tvs_pred_dict['TMCL'].append(tv[5])
            tvs_pred_dict['TMCD'].append(tv[6])
            tvs_pred_dict['TBCL'].append(tv[7])
            tvs_pred_dict['TBCD'].append(tv[8])

        # PHNS
        align_out = torch.max(att, axis=1)[1]
        pred_frame_phns = []
        for frame in align_out:
            pred_phn = int(phn_pred_seq[0][frame].detach().cpu().numpy())
            pred_frame_phns.append(pred_phn)
        
        return {
            'tvs_pred': tvs_pred_dict,
            'pred_frame_phns': pred_frame_phns,
            'pred_ctc_phn_seq': phn_pred_list,
            'hidden_alignment': att_out,
            'hidden_tvs': rnn_out[1],
        }

