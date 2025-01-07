import torch as t
import torch.nn as nn
import torchaudio

from transformers import (
    Wav2Vec2Model,
)
from utility import (
    idx_phonemes,
    phonemes_idx,
    convert_ts_float,
    get_children,
    _ctc_decode,
)



class Wav2Vec2_PR(t.nn.Module):
    """
    Wav2Vec2 model used as a phoneme recognizer.
    """
    def __init__(self, pretrain_cfg, cache_dir, huggingface_model_id, vocab):
        super().__init__()
        self.cache_dir = cache_dir
        self.huggingface_model_id = huggingface_model_id
        self.pretrain_cfg = pretrain_cfg
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            huggingface_model_id,
            config=pretrain_cfg,
            cache_dir=cache_dir,
        )
        self.wav2vec2.gradient_checkpointing_enable()
        
        self.dropout = nn.Dropout(pretrain_cfg.final_dropout)
        self.pr_head = nn.Linear(pretrain_cfg.hidden_size, pretrain_cfg.vocab_size)
        self.vocab = vocab


    def forward(
        self,
        input_values,
        input_lengths,
        phoneme_labels,
    ):
        # Wav2Vec2 output
        outputs = self.wav2vec2(
            input_values,
            attention_mask=input_lengths[:, None],
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Input
        state_lens = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        phoneme_logits = self.pr_head(hidden_states)
        log_probs = nn.functional.log_softmax(phoneme_logits, dim=-1, dtype=t.float32).transpose(0,1)

        # Targets
        targets = phoneme_labels
        target_lengths = []
        for phoneme_label in phoneme_labels:
            length = 0
            for label in phoneme_label:
                if label >= 0:
                    length += 1
            target_lengths.append(length)
        target_lengths = t.tensor(target_lengths)

        # Loss(es)
        loss = nn.functional.ctc_loss(
            log_probs,
            targets,
            state_lens,
            target_lengths,
            reduction=self.pretrain_cfg.ctc_loss_reduction,
            zero_infinity=self.pretrain_cfg.ctc_zero_infinity,
            blank = self.pretrain_cfg.blank
        )

        return ({
            'loss': loss,
            'phoneme_logits': phoneme_logits,
            'log_probs': log_probs,
            'hidden_states': hidden_states
        })


    def get_embeddings_grad(self, audio_inputs, audio_lengths, vocab, intermediate_hidden, latter_hidden):
        # feature extractor embedding: (batch, feat, time)
        features_hidden = self.wav2vec2.feature_extractor(audio_inputs)

        # phoneme embedding (batch, time, feat)
        w2v2_output = self.wav2vec2(
            audio_inputs,
            attention_mask=audio_lengths[:, None],
            return_dict=True,
            output_hidden_states=True,
        )
        last_transf_hidden = w2v2_output[0]
        # last_transf_hidden = w2v2_output.hidden_states[24]
        intermediate_hidden_emb = w2v2_output.hidden_states[intermediate_hidden]
        latter_hidden_emb = w2v2_output.hidden_states[latter_hidden]
        phoneme_logits_last = self.pr_head(last_transf_hidden)
        phoneme_logits_inter = self.pr_head(intermediate_hidden_emb)
        phoneme_logits_latter = self.pr_head(latter_hidden_emb)
        last_transf_hidden = last_transf_hidden.permute(0,2,1)
        intermediate_hidden_emb = intermediate_hidden_emb.permute(0,2,1)
        latter_hidden_emb = latter_hidden_emb.permute(0,2,1)

        return ({
            'features_hidden': features_hidden,
            'last_transf_hidden': last_transf_hidden,
            'phoneme_logits_last': phoneme_logits_last,
            'phoneme_logits_inter': phoneme_logits_inter,
            'phoneme_logits_latter': phoneme_logits_latter,
            'intermediate_hidden': intermediate_hidden_emb,
            'latter_hidden': latter_hidden_emb,
        })

    
    def get_embeddings(self, audio_inputs, audio_lengths):
        self.eval()
        device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        with t.no_grad():
            # feature extractor embedding: (batch, feat, time)
            features_hidden = self.wav2vec2.feature_extractor(audio_inputs)

            # phoneme embedding (batch, time, feat)
            w2v2_output = self.wav2vec2(
                audio_inputs,
                attention_mask=audio_lengths[:, None],
                return_dict=True,
                output_hidden_states=True,
            )
            last_transf_hidden = w2v2_output[0]
            phoneme_logits = self.pr_head(last_transf_hidden)
            frame_seq_lens = self.wav2vec2._get_feat_extract_output_lengths(audio_lengths)

            # ctc decoding
            vocab_list = [k for k,v in self.vocab.items()]
            beam_search_decoder = torchaudio.models.decoder.ctc_decoder(
                lexicon = None,
                tokens = vocab_list,
                lm = None,
                nbest = 1,
                beam_size = 10,
                beam_size_token = None,
                beam_threshold = 50,
                blank_token = '(blank)',
                sil_token = '(...)'
            )
            beam_search_result = beam_search_decoder(phoneme_logits.cpu())

            phn_seq_idx = []
            for result in beam_search_result:
                phn_seq_idx.append(result[0].tokens.detach().numpy())

            return ({
                'features_hidden': features_hidden,
                'last_transf_hidden': last_transf_hidden.permute(0,2,1),
                'phoneme_logits': phoneme_logits.cpu().numpy().transpose(0,2,1),
                'phn_pred_seq_idx': phn_seq_idx,
                'frame_seq_lens': frame_seq_lens.cpu().numpy(),
            })


    def get_ctc_logits(self, wav):
        self.eval()
        device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        with t.no_grad():
            if type(wav) is t.Tensor:
                wav = wav[0]
            wav_input = t.unsqueeze(t.Tensor(wav), dim=0).to(t.device(device))
            wav_len = t.unsqueeze(t.LongTensor([len(wav)]), dim=0).to(t.device(device))
            
            w2v2_out = self.wav2vec2(
                wav_input,
                attention_mask=wav_len,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = w2v2_out[0]
            phoneme_logits = self.pr_head(hidden_states)

            return phoneme_logits.squeeze(dim=0).detach().cpu().numpy()
        

    def predict_phonemes_durations(self, wav, vocab):
        self.eval()
        device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        with t.no_grad():
            if type(wav) is t.Tensor:
                wav = wav[0]
            wav_input = t.unsqueeze(t.Tensor(wav), dim=0).to(t.device(device))
            wav_len = t.unsqueeze(t.LongTensor([len(wav)]), dim=0).to(t.device(device))

            w2v2_out = self.wav2vec2(
                wav_input,
                attention_mask=wav_len,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = w2v2_out[0]
            phoneme_logits = self.pr_head(hidden_states)
            
            # ctc decoding
            vocab_list = [k for k,v in vocab.items()]
            frame_sec_ratio = wav.size(0) / phoneme_logits.size(1) / 16000
            beam_search_decoder = torchaudio.models.decoder.ctc_decoder(
                lexicon = None,
                tokens = vocab_list,
                lm = None,
                nbest = 1,
                beam_size = 10,
                beam_size_token = None,
                beam_threshold = 50,
                blank_token = '(blank)',
                sil_token = '(...)'
            )
            beam_search_result = beam_search_decoder(phoneme_logits.cpu())
            
            phon_seq_idx = beam_search_result[0][0].tokens.detach().numpy()
            phon_seq_ipa = idx_phonemes(vocab, phon_seq_idx)
            
            timestep_seq = beam_search_result[0][0].timesteps.detach().numpy()
            phon_seq_dur = [ts * frame_sec_ratio for ts in timestep_seq]
            
            return {
                'phn_seq_idx': phon_seq_idx,
                'phn_seq_ipa': phon_seq_ipa,
                'phn_seq_dur': phon_seq_dur,
            }


    def pred_phn_seq(self, wav, vocab):
        self.eval()
        device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        with t.no_grad():
            if type(wav) is t.Tensor:
                wav = wav[0]
            wav_input = t.unsqueeze(t.Tensor(wav), dim=0).to(t.device(device))
            wav_len = t.unsqueeze(t.LongTensor([len(wav)]), dim=0).to(t.device(device))

            w2v2_out = self.wav2vec2(
                wav_input,
                attention_mask=wav_len,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = w2v2_out[0]
            phoneme_logits = self.pr_head(hidden_states)
            
            # ctc decoding TODO call utils func here?
            vocab_list = [k for k,v in vocab.items()]
            frame_sec_ratio = wav.size(0) / phoneme_logits.size(1) / 16000
            beam_search_decoder = torchaudio.models.decoder.ctc_decoder(
                lexicon = None,
                tokens = vocab_list,
                lm = None,
                nbest = 1,
                beam_size = 10,
                beam_size_token = None,
                beam_threshold = 50,
                blank_token = '(blank)',
                sil_token = '(...)'
            )
            beam_search_result = beam_search_decoder(phoneme_logits.cpu())
            phon_seq_idx = beam_search_result[0][0].tokens.detach().numpy()
            phon_seq_ipa = idx_phonemes(vocab, phon_seq_idx)
            
            return {
                'phn_seq_idx': phon_seq_idx,
                'phn_seq_ipa': phon_seq_ipa,
            }


    
    
    def get_config(self):
        return {
            'huggingface_model_id': self.huggingface_model_id,
            'cache_dir': self.cache_dir,
            'pretrain_cfg': self.pretrain_cfg
        }


    def freeze_feature_encoder():
        self.wav2vec2.freeze_feature_encoder()
