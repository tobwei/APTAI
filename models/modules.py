import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import(
    pack_padded_sequence,
    pad_packed_sequence,
)
from torch import Tensor
import math


class LowPassFilterLayer(nn.Module):
    """
    Inpired from here:
    https://github.com/bootphon/articulatory_inversion/blob/master/Training/model.py
    """
    def __init__(self, device, cutoff, sampling_rate, out_dim=9):
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        self.filter_weights = self._get_filter_weights(cutoff, sampling_rate)
        self.filter_weights = self.filter_weights.view(1, 1, -1)
        self.lowpass = nn.Conv1d(1, self.out_dim, self.N, stride=1, padding='same', bias=False)
        self.lowpass.weight = nn.Parameter(self.filter_weights, requires_grad=False)

    def _get_filter_weights(self, cutoff, sampling_rate):
        """
        :return: low-pass filter weights
        """
        fc = cutoff / sampling_rate
        if fc > 0.5:
            raise Exception('Cutoff frequency must be at least twice the sampling rate.')
        b = 0.08 # transition band, as a fraction of the sampling rate (in(0, 0.5))
        N = int(np.ceil(4 / b)) # window
        if not N % 2:
            N += 1 # Make sure N is odd
        self.N = N
        n = np.arange(N)
        h = np.sinc(fc * 2 * (n - (N -1) / 2))
        w = 0.5 * (1 - np.cos(n * 2 * math.pi / (N - 1))) # compute hanning window
        h = h * w
        h = h / np.sum(h)
        return torch.tensor(h, device=self.device)

    def forward(self, y):
        """
        Apply convolution(i.e. lowpass filter) to each articulator separately.
        :param y: (B, L, dim_out) articulatory prediction not smoothed
        :return: smoothed y
        """
        y = y.double()
        B = len(y)
        L = len(y[0])
        y_smooth = torch.zeros(B, L, self.out_dim)
        for i in range(self.out_dim):
            traj_artic = y[:, :, i].view(B, 1, L)
            traj_artic_smooth = self.lowpass(traj_artic)
            traj_artic_smooth = traj_artic_smooth.view(B, L)
            y_smooth[:, :, i] = traj_artic_smooth
        return y_smooth.float().to(self.device)



class ForwardSumLoss(torch.nn.Module):
    '''
    Implementation from: https://nv-adlr.github.io/one-tts-alignment
    and https://github.com/lingjzhu/charsiu/blob/main/src/models.py
    '''
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
#        self.off_diag_penalty = off_diag_penalty
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)
        
    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq=target_seq.unsqueeze(0)
            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            
            # curr_logprob = curr_logprob + self.off_diagonal_prior(curr_logprob,text_lens[bid]+1,mel_lens[bid])
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
        # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total
    
    def off_diagonal_prior(self,log_prob,N, T, g=0.2): 
        n = torch.arange(N).to(log_prob.device)
        t = torch.arange(T).to(log_prob.device)
        t = t.unsqueeze(1).repeat(1,N)
        n = n.unsqueeze(0).repeat(T,1)
    
        W = torch.exp(-(n/N - t/T)**2/(2*g**2))
        return torch.log_softmax(W.unsqueeze(1),dim=-1)


class CrossAttention(nn.Module):
    '''
    Inspired by: https://github.com/lingjzhu/charsiu/blob/main/src/models.py
    '''
    def __init__(self, frame_dim, phn_dim, att_dim):
        super().__init__()
        self.q = nn.Linear(frame_dim, att_dim)
        self.k = nn.Linear(phn_dim, att_dim)
        self.layer_norm = nn.LayerNorm(att_dim*2)
        
    def forward(self, frame_hidden, phn_hidden, labels_att_mask):
        q_frame = self.q(frame_hidden)
        k_phn = self.k(phn_hidden)
        
        # energy shape: B, T, N (time in frames and N in sequence of phonemes)
        energy = torch.bmm(q_frame, k_phn.transpose(2, 1))
        att_mask = (1 - labels_att_mask) * - 1000.0
        energy = energy + att_mask.unsqueeze(1).repeat(1, energy.size(1), 1)

        att_matrix = torch.softmax(energy, dim=-1)
        att_out = torch.bmm(att_matrix, k_phn)
        att_out = torch.cat([att_out, q_frame], dim=-1)
        att_out = self.layer_norm(att_out)

        return att_out, energy


class ConvBank(nn.Module):
    '''
    Implementation from: https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/libri_phone/model.py
    '''
    def __init__(self, input_dim, output_class_num, kernels, cnn_size, hidden_size, dropout, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout
        
        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size, kernel, padding=kernel//2))
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.tanh(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:   
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.tanh(hidden), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted


class RNN(nn.Module):
    '''
    Adapted from:
    https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/libri_phone/model.py
    '''
    def __init__(self, hidden_dim, out_dim, drop=0.1):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim,hidden_dim,bidirectional=True,num_layers=1,batch_first=True)
        self.linear = nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),
                                    nn.Dropout(drop),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim,out_dim))
        
    def forward(self, embeddings, lens):
        if embeddings.shape[0] > 1:
            packed_input = pack_padded_sequence(embeddings, lens, batch_first=True,enforce_sorted=False)
            packed_output, (ht, ct)= self.lstm(packed_input)
            hidden_tvs = packed_putput
            out, _ = pad_packed_sequence(packed_output, batch_first=True)
            out = self.linear(out)
        elif embeddings.shape[0] == 1:
            out, (ht, ct) = self.lstm(embeddings)
            hidden_tvs = out
            out = self.linear(out)
        return out, hidden_tvs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


