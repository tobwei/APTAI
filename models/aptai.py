import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import(
    LowPassFilterLayer,
)
from transformers import (
    Wav2Vec2Model,
)


class APTAI(nn.Module):
    def __init__(
        self,
        device,
        vocab, 
        huggingface_model_id,
        pretrain_cfg,
        cache_dir,
        phn_drop = 0.1,
        tv_drop = 0.1,
        freeze_feature_encoder=True,
        ):
        super().__init__()
        self.device = device
        self.vocab = vocab
        self.huggingface_model_id = huggingface_model_id
        self.pretrain_cfg = pretrain_cfg
        
        # wav2vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            huggingface_model_id,
            config=pretrain_cfg,
            cache_dir=cache_dir,
        ).to(self.device)
        self.wav2vec2.gradient_checkpointing_enable()
        if freeze_feature_encoder:
            self.wav2vec2.freeze_feature_encoder()
        
        # TV head
        self.tv_head = nn.Sequential(
            nn.Dropout(tv_drop),
            nn.Tanh(),
            nn.Linear(1024, 9),
        )
        self.tv_lowpass = LowPassFilterLayer(self.device, 10, 49, 9)

        # phoneme head
        self.phn_head = nn.Sequential(
            nn.Dropout(phn_drop),
            nn.LeakyReLU(),
            nn.Linear(1024, 46),
        )
        
        
    def forward(
        self, epoch,
        audio_inputs, audio_lengths,
        phn_frames_49hz,
        LA, LP, JA,
        TTCL, TTCD,
        TMCL, TMCD,
        TBCL, TBCD,
        ):
        tv_targets = torch.stack([LA, LP, JA,
                              TTCL, TTCD,
                              TMCL, TMCD,
                              TBCL, TBCD], dim=-1).float()
        phn_targets = phn_frames_49hz
        tv_pad_mask = (tv_targets != -100.0)
        phn_pad_mask = (phn_frames_49hz != 0)
        
        w2v2_out = self.wav2vec2(
            audio_inputs,
            attention_mask=audio_lengths[:, None],
            return_dict=True,
            output_hidden_states=True,
        )
        last_transf_hidden = w2v2_out.hidden_states[24]

        tvs_out = self.tv_head(last_transf_hidden)
        tvs_out = self.tv_lowpass(tvs_out)

        phn_logits = self.phn_head(last_transf_hidden)
        
        # Loss
        mse_loss = F.mse_loss(
            input=tvs_out[tv_pad_mask],
            target=tv_targets[tv_pad_mask],
            reduction='mean',
        )
        # takes logits as input, indices as targets (padding with 0)
        ce_loss = F.cross_entropy(
            input=phn_logits.view(-1, phn_logits.size(2))[phn_pad_mask.flatten()],
            target=phn_targets.flatten()[phn_pad_mask.flatten()],
            ignore_index=0, 
            reduction='mean',
        )
        a = 0.5
        loss = a * mse_loss + (1 - a) * ce_loss
        
        # FC (frame classification) based phoneme prediction
        phn_probs = F.softmax(phn_logits, dim=-1)
        phn_fc_pred = torch.argmax(phn_probs, dim=-1)
        
        return {
            'loss': loss,
            'mse_loss': mse_loss,
            'ce_loss': ce_loss,
            'tvs_pred': tvs_out,
            'phn_fc_pred': phn_fc_pred,
            # 'phoneme_logits': phn_logits,
        }
    
    def get_config(self):
        return {
            'device': self.device,
            'vocab': self.vocab,
            'huggingface_model_id': self.huggingface_model_id,
            'pretrain_cfg': self.pretrain_cfg
        }

    def get_aptai_output(self, wav):
        self.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        with torch.no_grad():
            if type(wav) is torch.Tensor:
                wav = wav[0]
            wav_input = torch.unsqueeze(torch.Tensor(wav), dim=0).to(torch.device(device))
            wav_len = torch.unsqueeze(torch.LongTensor([len(wav)]), dim=0).to(torch.device(device))

            # w2v2 embeddings
            w2v2_out = self.wav2vec2(
                wav_input,
                attention_mask=wav_len[:, None],
                return_dict=True,
                output_hidden_states=True,
            )
            last_transf_hidden = w2v2_out.hidden_states[24]

            tvs_out = self.tv_head(last_transf_hidden)
            tvs_out = self.tv_lowpass(tvs_out)

            phn_logits = self.phn_head(last_transf_hidden)
            
            phn_probs = F.softmax(phn_logits, dim=-1)
            phn_fc_pred = torch.argmax(phn_probs, dim=-1)

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

            return {
                'phn_fc_probs': phn_probs.T.squeeze(dim=0).detach().cpu().numpy(),
                'phn_fc_logits': phn_logits.squeeze(dim=0).detach().cpu().numpy(),
                'phn_fc_pred': phn_fc_pred.squeeze(dim=0).detach().cpu().numpy(),
                'tvs_pred': tvs_pred_dict,
            }



