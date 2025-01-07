"""
Contains some utility functions.
"""
import os
import io
import pandas as pd
import editdistance
import csv
import requests
import xml.etree.ElementTree as et
import tqdm
import wandb
from transformers.integrations import is_wandb_available
from scipy import signal
from scipy.signal import get_window, butter, filtfilt
from scipy.stats import pearsonr
import librosa
from librosa.filters import mel
from librosa.sequence import dtw
import math
from itertools import groupby
import numpy as np
import pickle
import torch as t
import torchaudio
import torchaudio.models.decoder



def print_children(model: t.nn.Module):
    """
    Print childrens from a model (torch.nn.Module)
    """
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is:")
        print(child)
        child_counter += 1



def get_children(model: t.nn.Module):
    """
    Return childrens from a model (torch.nn.Module)
    """
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children



def count_parameters(model):
    """
    Counts and returns the parameters of a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    


def compute_mspec(wav_path):
    assert os.path.exists(wav_path)
    wav, fs = librosa.load(wav_path, sr=16000)

    mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    stft = pySTFT(wav).T
    stft_mel = np.dot(stft, mel_basis)
    stft_db = 20 * np.log10(np.maximum(min_level, stft_mel)) - 16
    mspec = (stft_db + 100) / 100        

    return mspec



def compute_PER(gt, pred):
    # same as in validate()
    edit_distance = editdistance.eval(gt, pred)
    num_phonemes = len(gt)
    per = edit_distance / num_phonemes
    return round(per * 100, 2)



def init_logger(cfg, project_name):
    """
    Used to init or resume a wandb logger.
    """
    if is_wandb_available():
        if(cfg.train_from_ckpt):
            print('wandb: resumes')
            runId = pickle.load(open(cfg.exp_dir / 'wandbRunId', 'rb'))
            wandb.init(
                id=runId,
                # see here: https://docs.wandb.ai/guides/runs/resuming
            )
        else:
            print('wandb: new run')
            runId = wandb.util.generate_id()
            pickle.dump(runId, open(cfg.exp_dir / 'wandbRunId', 'wb'))
            wandb.init(
                id=runId,
                project=project_name,
                name=cfg.exp_name,
                dir=cfg.exp_dir,
                entity='tobwei',
                resume='allow',
                config = {
                    'learning_rate': cfg.learning_rate,
                    'epochs': cfg.num_epochs,
                    'batch_size': cfg.batch_size
                },
            )
    else:
        print('\nLogging failed - wandb not available.\n')



def maus_g2p(audio_path, txt_path):
    """
    Returns: download link to phoneme textgrid result.

    Rest-API:
        https://clarin.phonetik.uni-muenchen.de/BASWebServices/help/help_developer#help_developer
        https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/help

    Speedup:
        Multi-Threading! https://realpython.com/intro-to-python-threading/
    """
    assert os.path.exists(audio_path)
    assert os.path.exists(txt_path)

    # This call returns a number (as string): 0 : low load, 1 : medium load, 2 : full load
    # Please do not issue more calls, when this call returns 2.
    server_status_url = 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/getLoadIndicator'
    status_res = requests.get(server_status_url)
    if status_res.status_code == 200:
        assert status_res.text != 2, 'Server load is too high.'

    url = 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runPipeline'
    with open(audio_path, 'rb') as a_f, open(txt_path, 'rb') as t_f:
        formdata = {
            'SIGNAL': a_f,
            'TEXT': t_f,
            'PIPE': (None, 'G2P_MAUS'), 
            'LANGUAGE': (None, 'eng'),
            'OUTFORMAT': (None, 'TextGrid'),
            'OUTSYMBOL': (None, 'ipa'),
            'USETEXTENHANCE': (None, 'false'), # causes error for single letters e.g. 'a'
            # 'OUT': (None, out_path)
        }
        res = requests.post(url, files=formdata)

    if res.status_code == 200:
        # print(res.text)
        tree = et.fromstring(res.text)
        download_link = tree.find('downloadLink').text
        # res2 = requests.get(tree.find('downloadLink').text)
        # if res2.status_code == 200:
        #     print('##### RESULT:')
        #     print('\t', res2.text)

    return download_link



def phon_ts_to_tuple(timestamps):
    """
    Takes a list of timestamps, and converts them to a 
    list of tuples:
    timestamps: [ts1, ts2, ...]
    timestampes_tuples: [(ts1, ts2), (ts2, ts3), ...]
    """
    raise NotImplementedError


def idx_phonemes(vocab, df):
    """
    maps input idx to phonemes based on the vocab.
    """
    key_list = list(vocab.keys())
    val_list = list(vocab.values())
    mapped_labels = []
    for idx in df:
        pos = val_list.index(int(idx))
        mapped_labels.append(key_list[pos])
    return mapped_labels


def idx_phn(phn_idx_seq, vocab):
    """
    maps input idx to phonemes based on the vocab.
    """
    key_list = list(vocab.keys())
    val_list = list(vocab.values())
    mapped_labels = []
    for idx in phn_idx_seq:
        pos = val_list.index(int(idx))
        mapped_labels.append(key_list[pos])
    return mapped_labels


def phn_idx(phn_seq, vocab):
    """
    maps input phn to phonemes based on the vocab.
    """
    mapped_labels = []
    for phn in phn_seq:
        mapped_labels.append(vocab[phn])
    return mapped_labels
    

def phonemes_idx(vocab, df):
    """
    takes the vocab and maps the phonemes of given input to idx of vocab
    """
    phoneme_labels = df.split(' ')
    mapped_labels = []
    for phoneme_label in phoneme_labels:
        mapped_labels.append(vocab[phoneme_label])
    return mapped_labels



def min_audio_duration(csv_path):
    """
    returns the minimum audio duration of a wav file contained in a corpus,
    which is provided via csv file.
    """
    assert os.path.exists(csv_path)

    min_duration = math.inf
    df = pd.read_csv(csv_path)

    for _, df_row in df.iterrows():
        audio, fs = torchaudio.load(df_row.path_wav)
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=fs, new_freq=16_000,
        )[0]

        duration = len(audio) / 16000

        if duration < min_duration:
            min_duration = duration

    return min_duration



def max_audio_duration(csv_path):
    """
    returns the maximum audio duration of a wav file contained in a corpus,
    which is provided via csv file.
    """
    assert os.path.exists(csv_path)

    max_duration = -math.inf
    df = pd.read_csv(csv_path)

    for _, df_row in df.iterrows():
        audio, fs = torchaudio.load(df_row.path_wav)
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=fs, new_freq=16_000,
        )[0]

        duration = len(audio) / 16000

        if duration > min_duration:
            max_duration = duration

    return max_duration



def convert_ts_float(input_string):
    """
    Used mainly for converting the phoneme_timestamps from the csv file back
    to float tupels when cropping.
    """
    input_string = input_string.replace('[', '').replace(']', '').replace(' ', '')
    tuple_strings = input_string.split('),(')
    float_tuples = []
    for tuple_string in tuple_strings:
        start, end = map(float, tuple_string.strip('()').split(','))
        float_tuples.append((start, end))
    return float_tuples


def match_phonemes_to_frames(
        phoneme_boundaries,
        phoneme_list,
        frame_duration=0.02
    ):
    matched_phonemes = []
    current_phoneme = None
    start = 0
    stop = int(phoneme_boundaries[-1] * 100) + 1
    step = int(frame_duration * 100)

    for frame_start in range(start, stop, step):
        frame_end = frame_start + int(frame_duration * 100)
        overlapping_phonemes = [phoneme for phoneme, boundary in zip(phoneme_list, phoneme_boundaries) 
            if frame_start / 100.0 <= boundary < frame_end / 100.0]

        if overlapping_phonemes:
            current_phoneme = overlapping_phonemes[0]
        matched_phonemes.append(current_phoneme)
        
        # if overlapping_phonemes:
        #     matched_phonemes.append(overlapping_phonemes[0])
        # else:
        #     matched_phonemes.append(None) # no match found for this frame

        # for boundary in phoneme_boundaries:
        #     if frame_start / 100 <= boundary < frame_end / 100.0:
        #         discretized_frames.append(boundary)
        #         break
            
    return matched_phonemes



def decode_textgrid_path(textgrid_path):
    grid = textgrids.TextGrid(textgrid_path)
    phoneme_labels = []
    phoneme_timestamps = []
    for mau in grid['MAU']:
        phoneme_labels.append(mau.text.transcode())
        phoneme_timestamps.append((mau.xmin, mau.xmax))
    return phoneme_labels, phoneme_timestamps



def decode_textgrid(textgrid):
    phoneme_labels = []
    phoneme_timestamps = []
    for mau in textgrid['MAU']:
        phoneme_labels.append(mau.text.transcode())
        phoneme_timestamps.append((mau.xmin, mau.xmax))
    return phoneme_labels, phoneme_timestamps



def plot_f0_wav(f0_rapt, wav, fs):
    # Create a time axis for the F0 values in seconds
    time_axis_f0 = np.arange(len(f0_rapt)) * 256 / fs
    # Create a time axis for the raw waveform in seconds
    time_axis_waveform = np.arange(len(wav)) / fs
    # Create a figure and an axis for the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot the F0 values on the first y-axis
    ax1.plot(time_axis_f0, f0_rapt, label='F0', color='red', marker='o')
    ax1.set_ylabel('F0 (Hz)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)
    # Create a second y-axis for the raw waveform
    ax2 = ax1.twinx()
    ax2.plot(time_axis_waveform, wav, color='blue', alpha=0.5)
    ax2.set_ylabel('Amplitude', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # Set x-axis label and title
    plt.xlabel('Time (seconds)')
    plt.title('Original Speech Signal with F0 Estimation')
    # Show the plot
    plt.grid(True)
    plt.show()



def tvs_metric_rmse(tvs_gt, tvs_pred):
    """
    tvs_gt and tvs_pred have the following order:
        LA, LP, JA, TTCL, TTCD, TMCL, TMCD, TBCL, TBCD
    """
    tvs_rmse_dict = {
        'LA': None,
        'LP': None,
        'JA': None,
        'TTCL': None,
        'TTCD': None,
        'TMCL': None,
        'TMCD': None,
        'TBCL': None,
        'TBCD': None,
    }
    tvs_gt_dict = {k: tvs_gt[:, i].tolist() for i,k in enumerate(tvs_rmse_dict.keys())}
    tvs_pred_dict = {k: tvs_pred[:, i].tolist() for i,k in enumerate(tvs_rmse_dict.keys())}

    for k,v in tvs_rmse_dict.items():
        se = np.square(np.subtract(tvs_gt_dict[k], tvs_pred_dict[k]))
        mse = sum(se) / len(se)
        rmse = math.sqrt(mse)
        tvs_rmse_dict[k] = rmse
        
    return tvs_rmse_dict



def tvs_metric_ppc(tvs_gt, tvs_pred):
    """
    tvs_gt and tvs_pred have the following order:
        LA, LP, JA, TTCL, TTCD, TMCL, TMCD, TBCL, TBCD
    """
    tvs_pcc_dict = {
        'LA': None,
        'LP': None,
        'JA': None,
        'TTCL': None,
        'TTCD': None,
        'TMCL': None,
        'TMCD': None,
        'TBCL': None,
        'TBCD': None,
    }
    tvs_gt_dict = {k: tvs_gt[:, i].tolist() for i,k in enumerate(tvs_pcc_dict.keys())}
    tvs_pred_dict = {k: tvs_pred[:, i].tolist() for i,k in enumerate(tvs_pcc_dict.keys())}

    for k,v in tvs_pcc_dict.items():
        tvs_pcc_dict[k] = pearsonr(tvs_gt_dict[k], tvs_pred_dict[k])
        
    return tvs_pcc_dict



def _ctc_decode(vocab, model_ouput):
    """
    Decodes given model output using given vocab using pytorch
    internal ctc decoding function.
    :param model_output: phoneme_logits
    """
    vocab_list = [k for k,v in vocab.items()]

    decoder = torchaudio.models.decoder.ctc_decoder(
        lexicon = None,
        # tokens = [(k, v) for k,v in vocab.items()], 
        # tokens = vocab,
        tokens = vocab_list,
        lm = None,
        nbest = 1,
        beam_size = 10,
        beam_size_token = None,
        beam_threshold = 50,
        blank_token = '(blank)',
        sil_token = '(...)'
    )
    decoded = decoder(t.from_numpy(model_ouput))
    decoded_tokens = decoded[0][0].tokens.detach().numpy()
    return decoded_tokens


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a dictionary with nested dictionaries by joining keys with a separator.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_csv(d, tgt_path, csv_name):
    """
    Writes a dictionary (possibly containing nested dictionaries) to a CSV file.
    """
    assert os.path.exists(tgt_path)
    csv_file_path = os.path.join(tgt_path, csv_name)
    
    flattened_data = flatten_dict(d)
    fieldnames = flattened_data.keys()

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(flattened_data)

        
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def interpolate_nan(signal):
    return pd.Series(signal).interpolate().tolist()



def force_align(cost, phn_ids) -> list:
    """
    Inspired by: 
    Force align text to audio.
    
    :param cost: float
    :param phn_ids: list of phoneme ids
    :returns: list of ids for aligend phonemes
    """
    D, align = dtw(C=-cost[:, phn_ids],
                   step_sizes_sigma=np.array([[1, 1], [1, 0]]))
    align_seq = [-1 for i in range(max(align[:, 0]) +1)]
    for i in list(align):
        print(align)
        if align_seq[i[0] < i[1]]:
            align_seq[i[0]] = i[1]

    align_ids = list(align_seq)
    return align_ids


def phn_frames2dur(phns, resolution=0.02):
    """
    Converts a sequence of phoneme labels based on frames
    to a list of tuples: (start, end, phn_id).
    
    :param phns: phoneme sequence (list)
    :param resolution: frame length
    :return: list of duration values
    """
    counter = 0 
    out = []
    for p, grp in groupby(phns):
        length = len(list(grp))
        out.append((round(counter * resolution, 2),
                    round((counter + length) * resolution, 2),
                    p
                    ))
        counter += length

    return out
        
    
def phn_frame_id2phn(frame_id_seq):
    phn_seq = []
    for p, grp in groupby(frame_id_seq):
        phn_seq.append(p)

    return phn_seq


"""
get_stats and get_metrics was adapted from unsupseg: https://github.com/felixkreuk/UnsupSeg
"""
def get_metrics(precision_counter, recall_counter, pred_counter, gt_counter):
    EPS = 1e-7
    eps = 1e-5

    precision = precision_counter / (pred_counter + eps)
    recall = recall_counter / (gt_counter + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    os = recall / (precision + EPS) - 1
    r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (np.sqrt(2))
    rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

    return precision, recall, f1, rval


def get_stats(y, yhat, tolerance=0.02):
    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0

    for yhat_i in yhat:
        min_dist = np.abs(y - yhat_i).min()
        precision_counter += (min_dist <= tolerance)
    
    for y_i in y:
        min_dist = np.abs(yhat - y_i).min()
        recall_counter += (min_dist <= tolerance)
    
    pred_counter += len(yhat)
    gt_counter += len(y)

    p, r, f1, rval = get_metrics(
        precision_counter,
        recall_counter,
        pred_counter,
        gt_counter
    )

    return p, r, f1, rval
        

def evaluate_overlap(gt_f, p_f):
    hits = 0
    counts = 0
    for targets,preds in zip(gt_f, p_f):
        assert len(targets)==len(preds)
        hits += sum(np.array(targets)==np.array(preds))
        counts += len(targets)
    return hits/counts



if __name__ == '__main__':
    # min_duration = min_audio_duration('../data/HPRC_prep/hprc.csv')
    # max_duration = max_audio_duration('../data/HPRC_prep/hprc.csv')
    # print('min_audio_duration', min_duration)
    # print('max_audio_duration', max_duration)

    # test = ctc2duration([1,1,1,2,2,2,2,3,3,4,4,5,27,18,18,18,7,7,1,1,1,1])
    # test = phn_frames2dur([1,2,15,17,23,32,41,1])
    # print(test)
    # input()
    


    print('\nmain done.')



