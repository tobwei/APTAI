
"""
Contains functions/classes around the HPRC dataset.
Important contents:
    - Pytorch Dataset
    - function to create hprc.csv
    - function to create the prepared version of HRPC
"""
import torch as t
import torchaudio
import os
import ast
import math
import csv
import pandas as pd
import random
import json
from sklearn import preprocessing as pre
import librosa
from librosa.filters import mel
import soundfile as sf
import textgrids
import pickle
import scipy
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    logging,
)
logging.set_verbosity_warning()
logging.set_verbosity_error()

from utility import (
    decode_textgrid,
    convert_ts_float,
    phonemes_idx,
    pySTFT,
    butter_lowpass_filter,
    interpolate_nan,
    match_phonemes_to_frames,
)
from w2v2_pr import (
    Wav2Vec2_PR,
)


class HPRCDataset(t.utils.data.Dataset):
    """
    EMA sensor and speech based dataset HPRC. 
    """
    def __init__(self, df, vocab, rate):
        assert rate in ['N', 'F', 'both']
        self.rate = rate
        self.vocab = vocab
        # change df depending on rate (select different data)
        if rate == 'N':
            self.df = df[df.rate == 'N']
        elif rate == 'F':
            self.df = df[df.rate == 'F']
        elif rate == 'both':
            self.df = df

    def __getitem__(self, index):
        # data
        row = self.df.iloc[index]

        # load audio + resample
        audio, fs = torchaudio.load(row.path_wav)
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=fs, new_freq=16_000,
        )[0]

        # len, phonemes
        audio_len = len(audio)
        phoneme_labels = phonemes_idx(self.vocab, row.phoneme_labels)
        phoneme_timestamps = row.phoneme_timestamps
        phoneme_timestamps = list(
            map(lambda x: float(x), phoneme_timestamps.strip('[]').split(', '))
        )

        # phonemes frames 49hz
        phn_frames_49hz = ast.literal_eval(row.phn_frames_49hz)

        # mspec
        # instead of computing it beforehand, could also do it live 
        # using the https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html
        mspec = pickle.load(open(row.path_mspec, 'rb'))
        mspec_len = len(mspec[1])

        # mfccs
        mfccs = pickle.load(open(row.path_mfccs, 'rb'))
        mfccs_len = len(mfccs[1])

        # speaker embedding
        spk_emb = spk_onehot_emb(row.speaker)

        # tvs
        file = open(row.path_tvs, 'rb')
        tvs = pickle.load(file)
        file.close()

        # tvs 49hz
        file = open(row.path_tvs_49hz, 'rb')
        tvs_49hz = pickle.load(file)
        file.close()

        # tvs_norm
        file = open(row.path_tvs_norm, 'rb')
        tvs_norm = pickle.load(file)
        file.close()

        # tvs_norm_49hz
        file = open(row.path_tvs_norm_49hz, 'rb')
        tvs_norm_49hz = pickle.load(file)
        file.close()

        return {
            # Input for forward
            'audio': audio,
            'audio_len': audio_len,
            'mspec': mspec,
            'mspec_len': mspec_len,
            'mfccs': mfccs,
            'mfccs_len': mfccs_len,
            'spk_emb': spk_emb,
            'phoneme_label': phoneme_labels,
            'phoneme_timestamps': phoneme_timestamps,
            'phn_frames_49hz': phn_frames_49hz,
            'tvs': tvs,
            'tvs_49hz': tvs_49hz,
            'tvs_norm': tvs_norm,
            'tvs_norm_49hz': tvs_norm_49hz,
        }

    def __len__(self):
        return len(self.df)



def hprc_csv(hprc_pre_path):
    """
    Selects one "F" and one "N" file (always first repetition if multiple are available)
    per utterance. 

    Note: it should be the pre-processed version of the hprc dataset.
    """
    assert os.path.exists(hprc_pre_path)

    data = []
    index = 0
    _, spk_dirs, _ = next(os.walk(hprc_pre_path))
    for spk_dir in spk_dirs:
        print(spk_dir)
        # base the data collection on the audio file
        audio_dir = os.path.join(hprc_pre_path, spk_dir, 'audio')
        tvs_dir = os.path.join(hprc_pre_path, spk_dir, 'tvs')
        tvs_49hz_dir = os.path.join(hprc_pre_path, spk_dir, 'tvs_49hz')
        tvs_norm_dir = os.path.join(hprc_pre_path, spk_dir, 'tvs_norm')
        tvs_norm_49hz_dir = os.path.join(hprc_pre_path, spk_dir, 'tvs_norm_49hz')
        mspec_dir = os.path.join(hprc_pre_path, spk_dir, 'mspec')
        mfccs_dir = os.path.join(hprc_pre_path, spk_dir, 'mfccs')
        _, _, audio_files = next(os.walk(audio_dir))
        for audio_file in audio_files:
            if '.wav' in audio_file and 'R01' in audio_file:
                data_dir = os.path.join(hprc_pre_path, spk_dir)
                file_name = audio_file[:-4]

                with open(os.path.join(data_dir, 'text', file_name + '.txt'), 'r') as tf:
                    text = tf.read().rstrip()

                phonemes = []
                phoneme_min_max = []
                grid = textgrids.TextGrid(
                    os.path.join(data_dir, 'phonemes', file_name + '.TextGrid')
                )
                for mau in grid['MAU']:
                    phonemes.append(mau.text.transcode())
                    phoneme_min_max.append((mau.xmin, mau.xmax))

                phoneme_timestamps = []
                for idx, ts_tuple in enumerate(phoneme_min_max):
                    if idx == len(phoneme_min_max) - 1:
                        # for the last one, also append the end
                        phoneme_timestamps.append(ts_tuple[0])
                        phoneme_timestamps.append(ts_tuple[1])
                    else:
                        phoneme_timestamps.append(ts_tuple[0])
                # Note: len(phoneme_timestamps) is one more than len(phonemes),
                # if want to convert to tuples, the middle value is used twice,
                # this is the reason for the len difference
                # Note2: previously phoneme_timestamples was list of tuples (0.0, 0.19),
                # (0.19, 0.39), ...
                
                phonemes = ' '.join(phonemes)

                # idx, path_wav, path_tvs, speaker, text, phonemes, phoneme_timestamps, rate
                # path_tvs, path_tvs_norm, path_tvs_norm_49hz, path_mspec
                row = []
                row.append(index)
                row.append(os.path.join(audio_dir, audio_file))
                row.append(spk_dir)
                row.append(text)
                row.append(phonemes)
                row.append(phoneme_timestamps)
                row.append(file_name[-1])
                row.append(os.path.join(tvs_dir, file_name + '.pkl'))
                row.append(os.path.join(tvs_49hz_dir, file_name + '.pkl'))
                row.append(os.path.join(tvs_norm_dir, file_name + '.pkl'))
                row.append(os.path.join(tvs_norm_49hz_dir, file_name + '.pkl'))
                row.append(os.path.join(mspec_dir, file_name + '.pkl'))
                row.append(os.path.join(mfccs_dir, file_name + '.pkl'))

                data.append(row)
                index += 1

    # save as hrpc.csv
    csv_path = os.path.join(hprc_pre_path, 'hprc.csv')
    columns = ['index', 'path_wav', 'speaker', 'text', 'phoneme_labels', 'phoneme_timestamps',
               'rate', 'path_tvs', 'path_tvs_49hz', 'path_tvs_norm', 'path_tvs_norm_49hz',
               'path_mspec', 'path_mfccs'] 
    with open(csv_path, 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerow(columns)
        write.writerows(data)



def hprc_processing(data_path, resample_fs=16000):
    """
    Creates a new directort 'hprc_pre' containing for each speaker:
        - Extracts the audio from the .mat to a .wav file, resampled to resample_fs.
        - Extracts the TextGrid grapheme/word transcriptons to a .txt file.
        - Extractes EMA data.
        - Creates Phoneme Labels using the MAUS webservice.
        - Creates F0.

    # EMA data:
        # TR   tongue rear (dorsum)
        # TB   tongue blade
        # TT   tongue tip (~ 1cm back from apex)
        # UL   upper lip (vermillion border)
        # LL   lower lip
        # ML   mouth left (corner)
        # JAW  jaw (lower medial incisors)
        # JAWL jaw left (canine)
    # EMA data orientation:
	   #  X - posterior -> anterior
	   #  Y - right -> left
	   #  Z - inferior -> superior
    # EMA trajectory format [nSamps x 6 dimensions]:
	   #  posX (mm)
	   #  posY
	   #  posZ
	   #  rotation around X (degrees)
	   #           around Y
	   #           around Z 
	# Palate
	    # one file per speaker
    """
    assert os.path.exists(data_path)
    target_path = data_path + '_prep'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    _, spk_dirs, _ = next(os.walk(data_path))
    for spk_dir in spk_dirs:
        print(spk_dir)
        _, sub_dirs, _ = next(os.walk(os.path.join(data_path, spk_dir)))
        for sub_dir in sub_dirs:
            # 1) audio & EMA data
            if 'data' in sub_dir:
                _, _, mat_files = next(os.walk(os.path.join(data_path, spk_dir, sub_dir)))
                for mat_file in mat_files:
                    if '.mat' in mat_file:
                        mat_file_path = os.path.join(data_path, spk_dir, sub_dir, mat_file)
                        mat_dict = scipy.io.loadmat(mat_file_path)
                        mat_data = mat_dict[mat_file[:-4]] # shape:(1, 9) - contains audio, trajectories

                        # audio: get from mat, resample, and save
                        # mat_data[0,X], with X= 0:audio, 1: TR, ....
                        if 'palate' not in mat_file:
                            fs = mat_data[0,0][1]
                            audio = mat_data[0,0][2]
                            new_audio_path = os.path.join(
                                target_path,
                                spk_dir,
                                'audio',
                            )
                            if not os.path.exists(new_audio_path):
                                os.makedirs(new_audio_path)
                            new_audio_file_path = os.path.join(
                                new_audio_path,
                                mat_file[:-4] + '.wav'
                            )
                            sf.write(new_audio_file_path, audio, fs)
                            tmp_audio, tmp_fs = sf.read(new_audio_file_path)
                            resampled_audio = librosa.resample(
                                tmp_audio,
                                orig_sr=tmp_fs,
                                target_sr=resample_fs
                            )
                            sf.write(new_audio_file_path, resampled_audio, resample_fs)
                        
                        # palate data (one file per speaker)
                        if 'palate' in mat_file:
                            palat_file_path = os.path.join(data_path, spk_dir, sub_dir, mat_file)
                            mat_dict = scipy.io.loadmat(palat_file_path)
                            palate_data = {
                                'x': [],
                                'y': [],
                                'z': []
                            }
                            spk_key = spk_dir + '_palate'
                            for row in mat_dict[spk_key]:
                                palate_data['x'].append(row[0])
                                palate_data['y'].append(row[1])
                                palate_data['z'].append(row[2])


                            new_ema_path = os.path.join(
                                target_path,
                                spk_dir,
                                'ema'
                            )
                            if not os.path.exists(new_ema_path):
                                os.makedirs(new_ema_path)
                            new_palate_file_path = os.path.join(
                                new_ema_path,
                                mat_file[:-4] + '.pkl'
                            )
                            file = open(new_palate_file_path, 'wb')
                            pickle.dump(palate_data, file)
                            file.close()

                        # EMA data 
                        # ML is not available for F02 -> not needed for TVs anyhow
                        if 'palate' not in mat_file:
                            ema_data = {
                                'TR': dict(),
                                'TB': dict(),
                                'TT': dict(),
                                'UL': dict(),
                                'LL': dict(),
                                'ML': dict(),
                                'JAW': dict(),
                                'JAWL': dict()
                            }
                            for i, ema_key in enumerate(ema_data.keys(), start=1):
                                # mat_data[0,X], with X= 1: TR, 2:TB, 3:TT, 4:UL, 5:LL, 6:ML, 7:JAW, 8:JAWL
                                if spk_dir == 'F02':
                                    # does not have ML data ...
                                    if ema_key == 'ML':
                                        continue
                                    if ema_key == 'JAW':
                                        all_data = mat_data[0,i-1][2]
                                    if ema_key == 'JAWL':
                                        all_data = mat_data[0,i-1][2]
                                    else:
                                        all_data = mat_data[0,i][2]

                                    tmp_dict = {
                                        'x': [],
                                        'y': [],
                                        'z': [],
                                    }
                                else:
                                    all_data = mat_data[0,i][2]
                                    tmp_dict = {
                                        'x': [],
                                        'y': [],
                                        'z': [],
                                        # 'r_x': [],
                                        # 'r_y': [],
                                        # 'r_z': []
                                    }
                                for row in all_data:
                                    # row= 0:x, 1:y, 2:z, 3:r_x, 4:r_y, 5:r_z
                                    tmp_dict['x'].append(row[0])
                                    tmp_dict['y'].append(row[1])
                                    tmp_dict['z'].append(row[2])
                                    # tmp_dict['r_x'].append(row[3])
                                    # tmp_dict['r_y'].append(row[4])
                                    # tmp_dict['r_z'].append(row[5])

                                ema_data[ema_key] = tmp_dict

                            new_ema_path = os.path.join(
                                target_path,
                                spk_dir,
                                'ema'
                            )
                            if not os.path.exists(new_ema_path):
                                os.makedirs(new_ema_path)
                            new_ema_file_path = os.path.join(
                                new_ema_path,
                                mat_file[:-4] + '.pkl'
                                
                            )
                            file = open(new_ema_file_path, 'wb')
                            pickle.dump(ema_data, file)
                            file.close()

            # 2) text transcipts
            if 'TextGrids' in sub_dir:
                # process TextGrids
                _, _, tg_files = next(os.walk(os.path.join(data_path, spk_dir, sub_dir)))
                for tg_file in tg_files:
                    if 'TextGrid' in tg_file:
                        word_list = []
                        # phone_list = []
                        tg_file_path = os.path.join(data_path, spk_dir, sub_dir, tg_file)
                        grid = textgrids.TextGrid(tg_file_path)
                        for w in grid['word']:
                            word_list.append(w.text.transcode())
                        # for p in grid['phone']:
                        #     phone_list.append(p.text.transcode())

                        clean_word_list = []
                        for w in word_list:
                            if w != 'sp':
                                clean_word_list.append(w)

                        txt = ' '.join(clean_word_list).lower()
                        new_txt_path = os.path.join(
                            target_path,
                            spk_dir,
                            'text',
                        )
                        if not os.path.exists(new_txt_path):
                            os.makedirs(new_txt_path)

                        txt_path = os.path.join(new_txt_path, tg_file[:-9] + '.txt')
                        with io.open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(txt)



def hprc_phoneme(root_dir, replace=False):
    """
    Required hprc_pre directory, based on which it will then create a subdir
    for each speaker containing phoneme TextGrids.
    """
    assert os.path.exists(root_dir)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_audio_dir = os.path.join(root_dir, spk_dir, 'audio')
        spk_txt_dir = os.path.join(root_dir, spk_dir, 'text')
        _, _, audio_files = next(os.walk(spk_audio_dir))
        num_files = 0
        num_spk_files = len(os.listdir(spk_audio_dir))

        spk_phoneme_dir = os.path.join(root_dir, spk_dir, 'phonemes')
        if not os.path.exists(spk_phoneme_dir):
            os.makedirs(spk_phoneme_dir)
        for audio_file in audio_files:
            target_path = os.path.join(spk_phoneme_dir, audio_file[:-4] + '.TextGrid')
            # check if phoneme file already exists
            if not replace and os.path.exists(target_path):
                num_files += 1
                print(f'(existed already) {spk_dir} -> {num_files}/{num_spk_files} files.')
                continue
            audio_path = os.path.join(spk_audio_dir, audio_file)
            txt_path = os.path.join(spk_txt_dir, audio_file[:-4] + '.txt')
            download_link = maus_g2p(audio_path, txt_path)
            res = requests.get(download_link, allow_redirects=True)
            open(target_path, 'wb').write(res.content)

            num_files += 1
            print(f'{spk_dir} -> {num_files}/{num_spk_files} files.')



def get_min_max_hprc_spk(root_dir, spk):
    """
    Return min_n, max_n, min_f, max_f, min_both, max_both
    for hprc utterances of the given speaker.

    For min-max the n and both are equal. 
    """
    assert os.path.exists(root_dir)
    
    tv_data_N = {
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
    tv_data_F = {
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
    tv_data_both = {
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
    
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        if spk_dir != spk:
            continue
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()
                        
                        # F
                        if tv_file_rate == 'F':
                            tv_data_F['LA'].extend(tvs_data['LA'])
                            tv_data_F['LP'].extend(tvs_data['LP'])
                            tv_data_F['JA'].extend(tvs_data['JA'])
                            tv_data_F['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_F['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_F['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_F['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_F['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_F['TBCD'].extend(tvs_data['TBCD'])
                            
                        # N
                        if tv_file_rate == 'N':
                            tv_data_N['LA'].extend(tvs_data['LA'])
                            tv_data_N['LP'].extend(tvs_data['LP'])
                            tv_data_N['JA'].extend(tvs_data['JA'])
                            tv_data_N['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_N['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_N['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_N['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_N['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_N['TBCD'].extend(tvs_data['TBCD'])
                            
                        # both
                        tv_data_both['LA'].extend(tvs_data['LA'])
                        tv_data_both['LP'].extend(tvs_data['LP'])
                        tv_data_both['JA'].extend(tvs_data['JA'])
                        tv_data_both['TTCL'].extend(tvs_data['TTCL'])
                        tv_data_both['TTCD'].extend(tvs_data['TTCD'])
                        tv_data_both['TMCL'].extend(tvs_data['TMCL'])
                        tv_data_both['TMCD'].extend(tvs_data['TMCD'])
                        tv_data_both['TBCL'].extend(tvs_data['TBCL'])
                        tv_data_both['TBCD'].extend(tvs_data['TBCD'])

    # compute mean/std per tv and for each rate, constrained by speaker
    min_max_tvs = {
        'LA': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'LP': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'JA': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TTCL': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TTCD': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TMCL': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TMCD': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TBCL': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TBCD': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
    }
    
    # min/max - N
    for key, value in tv_data_N.items():
        min = np.nanmin(np.array(tv_data_N[key]))
        max = np.nanmax(np.array(tv_data_N[key]))
        min_max_tvs[key]['min_n'] = min
        min_max_tvs[key]['max_n'] = max
                        
    # min/max - F
    for key, value in tv_data_F.items():
        min = np.nanmin(np.array(tv_data_F[key]))
        max = np.nanmax(np.array(tv_data_F[key]))
        min_max_tvs[key]['min_f'] = min
        min_max_tvs[key]['max_f'] = max

    # min/max - both
    for key, value in tv_data_both.items():
        min = np.nanmin(np.array(tv_data_both[key]))
        max = np.nanmax(np.array(tv_data_both[key]))
        min_max_tvs[key]['min_both'] = min
        min_max_tvs[key]['max_both'] = max

    return min_max_tvs



def get_mean_std_hprc_spk(root_dir, spk):
    """
    Return mean_n, std_n, mean_f std_f, mean_both, std_both
    for hprc utterances of the given speaker.
    """
    assert os.path.exists(root_dir)
    
    tv_data_N = {
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
    tv_data_F = {
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
    tv_data_both = {
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
    
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        if spk_dir != spk:
            continue
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()
                        
                        # F
                        if tv_file_rate == 'F':
                            tv_data_F['LA'].extend(tvs_data['LA'])
                            tv_data_F['LP'].extend(tvs_data['LP'])
                            tv_data_F['JA'].extend(tvs_data['JA'])
                            tv_data_F['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_F['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_F['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_F['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_F['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_F['TBCD'].extend(tvs_data['TBCD'])
                            
                        # N
                        if tv_file_rate == 'N':
                            tv_data_N['LA'].extend(tvs_data['LA'])
                            tv_data_N['LP'].extend(tvs_data['LP'])
                            tv_data_N['JA'].extend(tvs_data['JA'])
                            tv_data_N['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_N['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_N['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_N['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_N['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_N['TBCD'].extend(tvs_data['TBCD'])
                            
                        # both
                        tv_data_both['LA'].extend(tvs_data['LA'])
                        tv_data_both['LP'].extend(tvs_data['LP'])
                        tv_data_both['JA'].extend(tvs_data['JA'])
                        tv_data_both['TTCL'].extend(tvs_data['TTCL'])
                        tv_data_both['TTCD'].extend(tvs_data['TTCD'])
                        tv_data_both['TMCL'].extend(tvs_data['TMCL'])
                        tv_data_both['TMCD'].extend(tvs_data['TMCD'])
                        tv_data_both['TBCL'].extend(tvs_data['TBCL'])
                        tv_data_both['TBCD'].extend(tvs_data['TBCD'])

    # compute mean/std per tv and for each rate, constrained by speaker
    mean_std_tvs = {
        'LA': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'LP': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'JA': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TTCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TTCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TMCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TMCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TBCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TBCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
    }
    
    # mean/std - N
    for key, value in tv_data_N.items():
        mean = np.nanmean(np.array(tv_data_N[key]))
        std = np.nanstd(np.array(tv_data_N[key]))
        mean_std_tvs[key]['mean_n'] = mean
        mean_std_tvs[key]['std_n'] = std
                        
    # mean/std - F
    for key, value in tv_data_F.items():
        mean = np.nanmean(np.array(tv_data_F[key]))
        std = np.nanstd(np.array(tv_data_F[key]))
        mean_std_tvs[key]['mean_f'] = mean
        mean_std_tvs[key]['std_f'] = std

    # mean/std - both
    for key, value in tv_data_both.items():
        mean = np.nanmean(np.array(tv_data_both[key]))
        std = np.nanstd(np.array(tv_data_both[key]))
        mean_std_tvs[key]['mean_both'] = mean
        mean_std_tvs[key]['std_both'] = std

    return mean_std_tvs


def tvs_zscore_utterance(root_dir, rate):
    """
    Perform z-score normalization, on a utterance base,
    for all 'rate' utterances.
    """
    assert os.path.exists(root_dir)
    assert rate in ['F', 'N', 'both']
    
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        # only consider 'rate' files
                        if rate != 'both' and tv_file_rate != rate:
                            continue 
                        
                        # load tv data
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()

                        # zscore normalization per utterance
                        tvs_norm = {}
                        for key, tv_val in tvs_data.items():
                            mean = np.nanmean(np.array(tv_val))
                            std = np.nanstd(np.array(tv_val))
                            
                            centered = tv_val - mean
                            tvs_norm[key] = centered / std

                        # check if there is NaN, if so replace it
                        tvs_norm_tensor = t.Tensor(np.array(list(tvs_norm.values())))
                        nan_mask = t.isnan(tvs_norm_tensor)
                        assert len(nan_mask >= len(tvs_norm.values())/10)
                        tvs_norm_tensor[nan_mask] = 0.0
                        updated_values = tvs_norm_tensor.tolist()
                        tvs_norm = {k: v for k,v in zip(tvs_norm.keys(), updated_values)}

                        # store file
                        spk_tv_norm_dir = os.path.join(
                            root_dir,
                            spk_dir,
                            'tvs_norm'
                        )
                        if not os.path.exists(spk_tv_norm_dir):
                            os.makedirs(spk_tv_norm_dir)
                        spk_tv_norm_file = os.path.join(
                            spk_tv_norm_dir,
                            tv_file
                        )
                        pickle.dump(tvs_norm, open(spk_tv_norm_file, 'wb'))

    
    
def tvs_minmax_speaker(root_dir, rate):
    """
    Perform min-max normalization, on a speaker level,
    across 'rate' utterances.
    """
    assert os.path.exists(root_dir)
    assert rate in ['F', 'N', 'both']
    
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        min_max_spk_tvs = get_min_max_hprc_spk(root_dir, spk_dir)
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        # only consider 'rate' files
                        if rate != 'both' and tv_file_rate != rate:
                            continue 
                        
                        # load tv data
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()
                        
                        # min-max norm
                        tvs_norm = {
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
                        min_key = 'min_' + rate.lower()
                        max_key = 'max_' + rate.lower()
                        for tv, tv_values in tvs_data.items():
                            numerator = (tv_values - min_max_spk_tvs[tv][min_key]) * (1- -1)
                            denominator = min_max_spk_tvs[tv][max_key] - min_max_spk_tvs[tv][min_key]
                            tvs_norm[tv] = -1 + numerator / denominator

                        # store file
                        spk_tv_norm_dir = os.path.join(
                            root_dir,
                            spk_dir,
                            'tvs_norm'
                        )
                        if not os.path.exists(spk_tv_norm_dir):
                            os.makedirs(spk_tv_norm_dir)
                        spk_tv_norm_file = os.path.join(
                            spk_tv_norm_dir,
                            tv_file
                        )
                        pickle.dump(tvs_norm, open(spk_tv_norm_file, 'wb'))

                        

def tvs_zscore_speaker(root_dir, rate):
    """
    Perform z-score normalization, on a speaker level,
    across 'rate' utterances.
    """
    assert os.path.exists(root_dir)
    assert rate in ['F', 'N', 'both']
    
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        mean_std_spk_tvs = get_mean_std_hprc_spk(root_dir, spk_dir)
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        # only consider 'rate' files
                        if rate != 'both' and tv_file_rate != rate:
                            continue 

                        # load tv data
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()
                        
                        # z-scoring (speaker level)
                        tvs_norm = {
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
                        mean_key = 'mean_' + rate.lower()
                        std_key = 'std_' + rate.lower()
                        for tv, tv_values in tvs_data.items():
                            centered = tv_values - mean_std_spk_tvs[tv][mean_key]
                            tvs_norm[tv] = centered / mean_std_spk_tvs[tv][std_key]

                        # store file
                        spk_tv_norm_dir = os.path.join(
                            root_dir,
                            spk_dir,
                            'tvs_norm'
                        )
                        if not os.path.exists(spk_tv_norm_dir):
                            os.makedirs(spk_tv_norm_dir)
                        spk_tv_norm_file = os.path.join(
                            spk_tv_norm_dir,
                            tv_file
                        )
                        pickle.dump(tvs_norm, open(spk_tv_norm_file, 'wb'))
                        




def hprc_tvs_norm(root_dir, rate):
    """
    First performs a min-max scaling to a range between [-1, 1],
    followed by a zscore (0 mean, unit variance) normalization.

    Normalization is per TV and across all speakers, constrained
    by the rate.

    Requires hprc_pre and hprc_tvs() to have been run already.
    """
    assert os.path.exists(root_dir)
    assert rate in ['F', 'N', 'both']

    # 1) find min/max based mean and std value for zscoring
    min_max_tvs = get_min_max_hprc(root_dir)
    # base data for mean/std calcs
    tv_data_N = {
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
    tv_data_F = {
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
    tv_data_both = {
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

    # get all TV data, apply min-max norm 
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        assert os.path.exists(os.path.join(root_dir, spk_dir, 'tvs'))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()

                        # min/max norm
                        tvs_norm = {
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
                        min_key = 'min_' + rate.lower()
                        max_key = 'max_' + rate.lower()
                        for tv, tv_values in tvs_data.items():
                            numerator = (tv_values - min_max_tvs[tv][min_key]) * (1- -1)
                            denominator = min_max_tvs[tv][max_key] - min_max_tvs[tv][min_key]
                            tvs_norm[tv] = -1 + numerator / denominator
                            # tvs_norm_dict[tv] = -1 + ((tv_values - min_max_tvs[tv][min_key]) * (1- -1)) / (min_max_tvs[tv][max_key] - min_max_tvs[tv][min_key])
                        
                        # F
                        if tv_file_rate == 'F':
                            tv_data_F['LA'].extend(tvs_norm['LA'])
                            tv_data_F['LP'].extend(tvs_norm['LP'])
                            tv_data_F['JA'].extend(tvs_norm['JA'])
                            tv_data_F['TTCL'].extend(tvs_norm['TTCL'])
                            tv_data_F['TTCD'].extend(tvs_norm['TTCD'])
                            tv_data_F['TMCL'].extend(tvs_norm['TMCL'])
                            tv_data_F['TMCD'].extend(tvs_norm['TMCD'])
                            tv_data_F['TBCL'].extend(tvs_norm['TBCL'])
                            tv_data_F['TBCD'].extend(tvs_norm['TBCD'])

                        # N
                        if tv_file_rate == 'N':
                            tv_data_N['LA'].extend(tvs_norm['LA'])
                            tv_data_N['LP'].extend(tvs_norm['LP'])
                            tv_data_N['JA'].extend(tvs_norm['JA'])
                            tv_data_N['TTCL'].extend(tvs_norm['TTCL'])
                            tv_data_N['TTCD'].extend(tvs_norm['TTCD'])
                            tv_data_N['TMCL'].extend(tvs_norm['TMCL'])
                            tv_data_N['TMCD'].extend(tvs_norm['TMCD'])
                            tv_data_N['TBCL'].extend(tvs_norm['TBCL'])
                            tv_data_N['TBCD'].extend(tvs_norm['TBCD'])
                            
                        # both
                        tv_data_both['LA'].extend(tvs_norm['LA'])
                        tv_data_both['LP'].extend(tvs_norm['LP'])
                        tv_data_both['JA'].extend(tvs_norm['JA'])
                        tv_data_both['TTCL'].extend(tvs_norm['TTCL'])
                        tv_data_both['TTCD'].extend(tvs_norm['TTCD'])
                        tv_data_both['TMCL'].extend(tvs_norm['TMCL'])
                        tv_data_both['TMCD'].extend(tvs_norm['TMCD'])
                        tv_data_both['TBCL'].extend(tvs_norm['TBCL'])
                        tv_data_both['TBCD'].extend(tvs_norm['TBCD'])
    
    # perform actual normalization: 1) minmax 2) zscore 3) save
    mean_std_tvs = get_mean_std(tv_data_N, tv_data_F, tv_data_both)
    print(mean_std_tvs)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        assert os.path.exists(os.path.join(root_dir, spk_dir, 'tvs'))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()

                        tvs_norm = {
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
                        plt.plot(tvs_data['LA'], color='red')
                        plt.show()
                        # 1) min-max norm
                        min_key = 'min_' + rate.lower()
                        max_key = 'max_' + rate.lower()
                        for tv, tv_values in tvs_data.items():
                            numerator = (tv_values - min_max_tvs[tv][min_key]) * (1- -1)
                            denominator = min_max_tvs[tv][max_key] - min_max_tvs[tv][min_key]
                            tvs_norm[tv] = -1 + numerator / denominator
                            
                        plt.plot(tvs_norm['LA'], color='blue')
                        plt.show()
                        # 2) zscore norm based on 1)
                        mean_key = 'mean_' + rate.lower()
                        std_key = 'std_' + rate.lower()
                        for tv, tv_values in tvs_norm.items():
                            centered = (tv_values - mean_std_tvs[tv][mean_key])
                            tvs_norm[tv] = centered / mean_std_tvs[tv][std_key]

                        plt.plot(tvs_norm['LA'], color='green')
                        plt.show()

                        # 3) save
                        input()





    

    #
    #                     # store data
    #                     spk_tv_norm_dir = os.path.join(
    #                         root_dir,
    #                         spk_dir,
    #                         'tvs_norm'
    #                     )
    #                     if not os.path.exists(spk_tv_norm_dir):
    #                         os.makedirs(spk_tv_norm_dir)
    #                     spk_tv_norm_file = os.path.join(
    #                         spk_tv_norm_dir,
    #                         tv_file
    #                     )
    #                     file = open(spk_tv_norm_file, 'wb')
    #                     pickle.dump(tvs_norm_dict, file)
    #                     file.close()
    





    input()

    


    


    # TODO implement the rate 
def hprc_tvs_norm_zscore(root_dir, rate):
    """
    Tvs are normalized across all speakers, taking all
    utterances of the specificed 'rate' parameter into
    consideration.
    
    Requires hprc_pre and hprc_tvs() to have been run already.
    """ 

    assert os.path.exists(root_dir)
    assert rate in ['F', 'N', 'both']
    # TODO -> rate in get_mean_std_hprc
    mean_std_tvs = get_mean_std_hprc(root_dir)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file and 'palate' not in tv_file:
                        # load required data
                        tvs = open(os.path.join(spk_tv_dir, tv_file), 'rb')
                        tvs_dict = pickle.load(tvs)
                        tvs_norm_dict = {
                            'LA': [],
                            'LP': [],
                            'JA': [],
                            'TTCL': [],
                            'TTCD': [],
                            'TMCL': [],
                            'TMCD': [],
                            'TBCL': [],
                            'TBCD': []
                        }
                        tvs.close()

                        # standardize data
                        for tv, tv_values in tvs_dict.items():
                            centered = (tv_values - mean_std_tvs[tv]['mean'])
                            tvs_norm_dict[tv] = centered / mean_std_tvs[tv]['std']

                        # store data
                        spk_tv_norm_dir = os.path.join(
                            root_dir,
                            spk_dir,
                            'tvs_norm'
                        )
                        if not os.path.exists(spk_tv_norm_dir):
                            os.makedirs(spk_tv_norm_dir)
                        spk_tv_norm_file = os.path.join(
                            spk_tv_norm_dir,
                            tv_file
                        )
                        file = open(spk_tv_norm_file, 'wb')
                        pickle.dump(tvs_norm_dict, file)
                        file.close()




def hprc_tvs(root_dir: str, lowpass: bool = True):
    """
    Required hprc_pre directory, based on which it will then create a subdir
    for each speaker containing the tv's as pickle file (dict).
    If lowpass == True, takes 'ema_low' dir, otherwise 'ema'.
    """
    assert os.path.exists(root_dir)
    ema = 'ema_low' if lowpass else 'ema'

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_ema_dir = os.path.join(root_dir, spk_dir, ema)
        # one palate file per speaker
        palate_file_path = os.path.join(spk_ema_dir, spk_dir + '_palate.pkl')
        palate_file = open(palate_file_path, 'rb')
        palate_data = pickle.load(palate_file)
        palate_file.close()
        # speaker based median values
        median_LLx = get_median_ema_x(spk_ema_dir, 'LL')
        median_TTx = get_median_ema_x(spk_ema_dir, 'TT')
        median_TBx = get_median_ema_x(spk_ema_dir, 'TB')
        median_TRx = get_median_ema_x(spk_ema_dir, 'TR')
        # other ema files per speaker and per audio file
        _, _, spk_ema_files = next(os.walk(spk_ema_dir))
        for spk_ema_file in spk_ema_files:
            if 'palate' not in spk_ema_file:
                ema_file_path = os.path.join(spk_ema_dir, spk_ema_file)
                ema_file = open(ema_file_path, 'rb')
                ema_data = pickle.load(ema_file)
                ema_file.close()
                ema_length = len(ema_data['TR']['x']) # all ema's and x,y,z have same length

                # compute 9 TVs
                tvs_dict = {
                    'LA': [],
                    'LP': [],
                    'JA': [],
                    'TTCL': [],
                    'TTCD': [],
                    'TMCL': [],
                    'TMCD': [],
                    'TBCL': [],
                    'TBCD': []
                }
                for n in range(0, ema_length):
                    # LA
                    la_n = math.sqrt(
                        (ema_data['LL']['x'][n] - ema_data['UL']['x'][n])**2 +
                        (ema_data['LL']['z'][n] - ema_data['UL']['z'][n])**2
                    )
                    tvs_dict['LA'].append(la_n)
                    # LP
                    lp_n = ema_data['LL']['x'][n] - median_LLx
                    tvs_dict['LP'].append(lp_n)
                    # JA
                    ja_n = math.sqrt(
                        (ema_data['JAW']['x'][n] - ema_data['UL']['x'][n])**2 +
                        (ema_data['JAW']['z'][n] - ema_data['UL']['z'][n])**2
                    )
                    tvs_dict['JA'].append(ja_n)
                    # TTCL
                    ttcl_n = median_TTx - ema_data['TT']['x'][n]
                    tvs_dict['TTCL'].append(ttcl_n)
                    # TMCL
                    tmcl_n = median_TBx - ema_data['TB']['x'][n]
                    tvs_dict['TMCL'].append(tmcl_n)
                    # TBCL
                    tbcl_n = median_TRx - ema_data['TR']['x'][n]
                    tvs_dict['TBCL'].append(tbcl_n)
                    # TTCD
                    ttcd_candidates = []
                    for x in range(-50, 0):
                        ttcd_candidates.append(
                            math.sqrt(
                                (ema_data['TT']['x'][n] - x)**2 +
                                (ema_data['TT']['z'][n] - palate_data['z'][(-x)-1])**2
                            )
                        )
                    ttcd_n = np.min(ttcd_candidates)
                    tvs_dict['TTCD'].append(ttcd_n)
                    # TMCD
                    tmcd_candidates = []
                    for x in range(-50, 0):
                        tmcd_candidates.append(
                            math.sqrt(
                                (ema_data['TB']['x'][n] - x)**2 +
                                (ema_data['TB']['z'][n] - palate_data['z'][(-x)-1])**2
                            )
                        )
                    tmcd_n = np.min(tmcd_candidates)
                    tvs_dict['TMCD'].append(tmcd_n)
                    # TBCD
                    tbcd_candidates = []
                    for x in range(-50, 0):
                        tbcd_candidates.append(
                            math.sqrt(
                                (ema_data['TR']['x'][n] - x)**2 +
                                (ema_data['TR']['z'][n] - palate_data['z'][(-x)-1])**2
                            )
                        )
                    tbcd_n = np.min(tbcd_candidates)
                    tvs_dict['TBCD'].append(tbcd_n)
            # save 
            tv_dir_path = os.path.join(
                root_dir,
                spk_dir,
                'tvs'
            )
            if not os.path.exists(tv_dir_path):
                os.makedirs(tv_dir_path)
            tv_file_path = os.path.join(
                tv_dir_path,
                spk_ema_file
            )
            file = open(tv_file_path, 'wb')
            pickle.dump(tvs_dict, file)
            file.close()



def get_median_ema_x(spk_ema_dir, ema):
    """
    Returns the median across all utterances of a given speaker
    in terms of the given ema value. This is required for some
    TV caluclations.
    """
    assert os.path.exists(spk_ema_dir)
    assert ema in ['LL', 'TT', 'TB', 'TR']
    _, _, spk_ema_files = next(os.walk(spk_ema_dir))
    file_medians = []
    for ema_file in spk_ema_files:
        if 'palate' not in ema_file:
            ema_file_path = os.path.join(spk_ema_dir, ema_file)
            ema_file = open(ema_file_path, 'rb')
            ema_data = pickle.load(ema_file)
            ema_file.close()

            file_medians.append(np.nanmedian(ema_data[ema]['x']))
    return np.median(file_medians)



def get_mean_std(tv_data_N, tv_data_F, tv_data_both):
    """
    tv_dict has all values across speakers for each TV.
    """
    # return dict
    mean_std_tvs = {
        'LA': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'LP': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'JA': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TTCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TTCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TMCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TMCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TBCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TBCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
    }
    
    # mean/std - N
    for key, value in tv_data_N.items():
        mean = np.nanmean(np.array(tv_data_N[key]))
        std = np.nanstd(np.array(tv_data_N[key]))
        mean_std_tvs[key]['mean_n'] = mean
        mean_std_tvs[key]['std_n'] = std
                        
    # mean/std - F
    for key, value in tv_data_F.items():
        mean = np.nanmean(np.array(tv_data_F[key]))
        std = np.nanstd(np.array(tv_data_F[key]))
        mean_std_tvs[key]['mean_f'] = mean
        mean_std_tvs[key]['std_f'] = std

    # mean/std - both
    for key, value in tv_data_both.items():
        mean = np.nanmean(np.array(tv_data_both[key]))
        std = np.nanstd(np.array(tv_data_both[key]))
        mean_std_tvs[key]['mean_both'] = mean
        mean_std_tvs[key]['std_both'] = std

    return mean_std_tvs



def get_min_max_hprc(root_dir):
    """
    Based on original tv data (unormalized).
    """
    assert os.path.exists(root_dir)
    # return dict
    min_max_tvs = {
        'LA': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'LP': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'JA': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TTCL': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TTCD': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TMCL': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TMCD': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TBCL': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
        'TBCD': {'min_n': 0, 'max_n': 0,
               'min_f': 0, 'max_f': 0,
               'min_both': 0, 'max_both': 0},
    }
    # base data for min/max calc
    tv_data_N = {
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
    tv_data_F = {
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
    tv_data_both = {
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
    
    # get all TV data across hprc speakers
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        assert os.path.exists(os.path.join(root_dir, spk_dir, 'tvs'))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()
                        
                        # F
                        if tv_file_rate == 'F':
                            tv_data_F['LA'].extend(tvs_data['LA'])
                            tv_data_F['LP'].extend(tvs_data['LP'])
                            tv_data_F['JA'].extend(tvs_data['JA'])
                            tv_data_F['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_F['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_F['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_F['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_F['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_F['TBCD'].extend(tvs_data['TBCD'])

                        # N
                        if tv_file_rate == 'N':
                            tv_data_N['LA'].extend(tvs_data['LA'])
                            tv_data_N['LP'].extend(tvs_data['LP'])
                            tv_data_N['JA'].extend(tvs_data['JA'])
                            tv_data_N['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_N['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_N['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_N['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_N['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_N['TBCD'].extend(tvs_data['TBCD'])
                            
                        # both
                        tv_data_both['LA'].extend(tvs_data['LA'])
                        tv_data_both['LP'].extend(tvs_data['LP'])
                        tv_data_both['JA'].extend(tvs_data['JA'])
                        tv_data_both['TTCL'].extend(tvs_data['TTCL'])
                        tv_data_both['TTCD'].extend(tvs_data['TTCD'])
                        tv_data_both['TMCL'].extend(tvs_data['TMCL'])
                        tv_data_both['TMCD'].extend(tvs_data['TMCD'])
                        tv_data_both['TBCL'].extend(tvs_data['TBCL'])
                        tv_data_both['TBCD'].extend(tvs_data['TBCD'])

    # min/max - N
    for key, value in tv_data_N.items():
        min = np.nanmin(np.array(tv_data_N[key]))
        max = np.nanmax(np.array(tv_data_N[key]))
        min_max_tvs[key]['min_n'] = min
        min_max_tvs[key]['max_n'] = max
                        
    # min/max - F
    for key, value in tv_data_F.items():
        min = np.nanmin(np.array(tv_data_F[key]))
        max = np.nanmax(np.array(tv_data_F[key]))
        min_max_tvs[key]['min_f'] = min
        min_max_tvs[key]['max_f'] = max

    # min/max - both
    for key, value in tv_data_both.items():
        min = np.nanmin(np.array(tv_data_both[key]))
        max = np.nanmax(np.array(tv_data_both[key]))
        min_max_tvs[key]['min_both'] = min
        min_max_tvs[key]['max_both'] = max

    return min_max_tvs



def get_mean_std_hprc(root_dir):
    """
    Returns a dict that conaints the median and std for:
    N, F, and both speaking rates.

    Required for normalization (0 mean, 1std) for each TV,
    across all speakers.
    Returns mean and std for each TV across all speakers.
    Process:
        for each TV, create a giant list across all speakers
        calculate mean of this list -> store in dict
        calculate std of this list -> store in dict
        return dict
    """
    assert os.path.exists(root_dir)
    # return dict
    mean_std_tvs = {
        'LA': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'LP': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'JA': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TTCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TTCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TMCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TMCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TBCL': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
        'TBCD': {'mean_n': 0, 'std_n': 0,
               'mean_f': 0, 'std_f': 0,
               'mean_both': 0, 'std_both': 0},
    }
    # base data for median/std calculation
    tv_data_N = {
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
    tv_data_F = {
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
    tv_data_both = {
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

    # get all TV data across hprc speakers
    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        _, sub_dirs, _ = next(os.walk(os.path.join(root_dir, spk_dir)))
        assert os.path.exists(os.path.join(root_dir, spk_dir, 'tvs'))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                spk_tv_dir = os.path.join(root_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(spk_tv_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file:
                        tv_file_rate = tv_file[:-4][-1]
                        tv_file_path = os.path.join(spk_tv_dir, tv_file)
                        tvs = open(tv_file_path, 'rb')
                        tvs_data = pickle.load(tvs)
                        tvs.close()
                        
                        # F
                        if tv_file_rate == 'F':
                            tv_data_F['LA'].extend(tvs_data['LA'])
                            tv_data_F['LP'].extend(tvs_data['LP'])
                            tv_data_F['JA'].extend(tvs_data['JA'])
                            tv_data_F['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_F['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_F['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_F['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_F['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_F['TBCD'].extend(tvs_data['TBCD'])

                        # N
                        if tv_file_rate == 'N':
                            tv_data_N['LA'].extend(tvs_data['LA'])
                            tv_data_N['LP'].extend(tvs_data['LP'])
                            tv_data_N['JA'].extend(tvs_data['JA'])
                            tv_data_N['TTCL'].extend(tvs_data['TTCL'])
                            tv_data_N['TTCD'].extend(tvs_data['TTCD'])
                            tv_data_N['TMCL'].extend(tvs_data['TMCL'])
                            tv_data_N['TMCD'].extend(tvs_data['TMCD'])
                            tv_data_N['TBCL'].extend(tvs_data['TBCL'])
                            tv_data_N['TBCD'].extend(tvs_data['TBCD'])
                            
                        # both
                        tv_data_both['LA'].extend(tvs_data['LA'])
                        tv_data_both['LP'].extend(tvs_data['LP'])
                        tv_data_both['JA'].extend(tvs_data['JA'])
                        tv_data_both['TTCL'].extend(tvs_data['TTCL'])
                        tv_data_both['TTCD'].extend(tvs_data['TTCD'])
                        tv_data_both['TMCL'].extend(tvs_data['TMCL'])
                        tv_data_both['TMCD'].extend(tvs_data['TMCD'])
                        tv_data_both['TBCL'].extend(tvs_data['TBCL'])
                        tv_data_both['TBCD'].extend(tvs_data['TBCD'])

    # mean/std - N
    for key, value in tv_data_N.items():
        mean = np.nanmean(np.array(tv_data_N[key]))
        std = np.nanstd(np.array(tv_data_N[key]))
        mean_std_tvs[key]['mean_n'] = mean
        mean_std_tvs[key]['std_n'] = std
                        
    # mean/std - F
    for key, value in tv_data_F.items():
        mean = np.nanmean(np.array(tv_data_F[key]))
        std = np.nanstd(np.array(tv_data_F[key]))
        mean_std_tvs[key]['mean_f'] = mean
        mean_std_tvs[key]['std_f'] = std

    # mean/std - both
    for key, value in tv_data_both.items():
        mean = np.nanmean(np.array(tv_data_both[key]))
        std = np.nanstd(np.array(tv_data_both[key]))
        mean_std_tvs[key]['mean_both'] = mean
        mean_std_tvs[key]['std_both'] = std

    return mean_std_tvs



def hprc_mspec_znorm(root_dir):
    """
    Utterance based z-score normalization.
    """
    assert os.path.exists(root_dir)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_mspec_dir = os.path.join(root_dir, spk_dir, 'mspec')
        _, _, spk_mspec_files = next(os.walk(spk_mspec_dir))
        for spk_mspec_file in spk_mspec_files:
            # load
            mspec_path = os.path.join(spk_mspec_dir, spk_mspec_file)
            mspec = pickle.load(open(mspec_path, 'rb'))
            
            # zscore norm utterance based
            mean = np.mean(mspec, axis=0)
            std = np.std(mspec, axis=0)
            norm_mspec = (mspec - mean) / std

            # sr = 16000
            # target_time_resolution = 49
            # hop_length = int(np.round(sr / target_time_resolution))
            # librosa.display.specshow(mspec, x_axis='time',
            #                          y_axis='mel', sr=sr, hop_length=hop_length)
            # librosa.display.specshow(norm_mspec, x_axis='time',
            #                          y_axis='mel', sr=sr, hop_length=hop_length)
            # plt.show()

            # save mspec (overwrite not-normalized version)
            mspec_dir_path = os.path.join(
                root_dir,
                spk_dir,
                'mspec'
            )
            if not os.path.exists(mspec_dir_path):
                os.makedirs(mspec_dir_path)
            mspec_file_path = os.path.join(
                mspec_dir_path,
                spk_audio_file[:-4] + '.pkl'

            )
            pickle.dump(norm_mspec, open(mspec_file_path, 'wb'))


def hprc_mfccs(root_dir, n_mfccs=13):
    """
    MFCC for the HPRC dataset, based on 49hz.
    Uses same values for the log-power mel spectrogram as hprc_mspec.
    Could also load and use the already computed mspecs.
    """
    assert os.path.exists(root_dir)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_audio_dir = os.path.join(root_dir, spk_dir, 'audio')
        _, _, spk_audio_files = next(os.walk(spk_audio_dir))
        for spk_audio_file in spk_audio_files:
            # load audio
            sr = 16000
            target_time_resolution = 49
            # hop_length = int(np.round(sr / target_time_resolution))
            hop_length = int(sr / target_time_resolution) - 4
            y, sr = librosa.load(os.path.join(spk_audio_dir, spk_audio_file), sr=sr)

            # 1) 
            mfccs = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_fft=1024,
                hop_length=hop_length,
                n_mfcc=n_mfccs,
                fmin=90,
                fmax=7600,
                center=True,
                pad_mode='constant'
            )
            
            # save mfccs
            mfccs_dir_path = os.path.join(
                root_dir,
                spk_dir,
                'mfccs'
            )
            if not os.path.exists(mfccs_dir_path):
                os.makedirs(mfccs_dir_path)
            mfccs_file_path = os.path.join(
                mfccs_dir_path,
                spk_audio_file[:-4] + '.pkl'

            )
            pickle.dump(mfccs, open(mfccs_file_path, 'wb'))

            

def hprc_mspec(root_dir):
    """
    mel-spectorgram (db) based on 49hz. 
    """
    assert os.path.exists(root_dir)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_audio_dir = os.path.join(root_dir, spk_dir, 'audio')
        _, _, spk_audio_files = next(os.walk(spk_audio_dir))
        for spk_audio_file in spk_audio_files:
            # load audio
            sr = 16000
            target_time_resolution = 49
            # hop_length = int(np.round(sr / target_time_resolution))
            hop_length = int(sr / target_time_resolution) - 4
            y, sr = librosa.load(os.path.join(spk_audio_dir, spk_audio_file), sr=sr)
            
            # 1) 
            power_mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=1024,
                hop_length=hop_length,
                n_mels=128,
                fmin=90,
                fmax=7600,
                center=True,
                pad_mode='constant'
            )
            mspec = librosa.power_to_db(power_mel_spec, ref=np.max)

            # 2) 
            # mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
            # min_level = np.exp(-100 / 20 * np.log(10))
            # stft = pySTFT(y, fft_length=1024, hop_length=hop_length).T
            # stft_mel = np.dot(stft, mel_basis)
            # stft_db = 20 * np.log10(np.maximum(min_level, stft_mel)) - 16
            # mspec = (stft_db + 100) / 100        

            # librosa.display.specshow(mspec, x_axis='time',
            #                          y_axis='mel', sr=sr, hop_length=hop_length)
            # plt.imshow(mspec)
            # plt.show()
            
            # save mspec
            mspec_dir_path = os.path.join(
                root_dir,
                spk_dir,
                'mspec'
            )
            if not os.path.exists(mspec_dir_path):
                os.makedirs(mspec_dir_path)
            mspec_file_path = os.path.join(
                mspec_dir_path,
                spk_audio_file[:-4] + '.pkl'

            )
            pickle.dump(mspec, open(mspec_file_path, 'wb'))


def hprc_w2v2_phn_embs(root_dir, model):
    """
    Extracts the w2v2 phoneme embeddings for the hprc utterances,
    this is mainly to speed up training if w2v2 layers are frozen.
    """
    assert os.path.exists(root_dir)


    # iterate over spk_stuff and load the wav's 

    # use the get embedding without grad from the model 

    # store the embedding in its own dir


    input()




    

def hprc_f0_mspec(root_dir):
    """
    https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.rapt.html?highlight=rapt
    """
    assert os.path.exists(root_dir)

    _, spk_dirs, _ = next(os.walk(root_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        if 'M' in spk_dir:
            lo, hi = 50, 250
        elif 'F' in spk_dir:
            lo, hi = 100, 600
        else:
            raise ValueError
        
        spk_audio_dir = os.path.join(root_dir, spk_dir, 'audio')
        _, _, spk_audio_files = next(os.walk(spk_audio_dir))
        for spk_audio_file in spk_audio_files:
            # load audio
            wav, fs = sf.read(os.path.join(spk_audio_dir, spk_audio_file))
            assert fs == 16000, 'Signal has to be sampled at 16kHz.'

            # f0
            f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=1)
            # plot_f0_wav(f0_rapt, wav, fs)
            # save f0
            # input()
            f0_dir_path = os.path.join(
                root_dir,
                spk_dir,
                'f0'
            )
            if not os.path.exists(f0_dir_path):
                os.makedirs(f0_dir_path)
            f0_file_path = os.path.join(
                f0_dir_path,
                spk_audio_file[:-4] + '.pkl'

            )
            file = open(f0_file_path, 'wb')
            pickle.dump(f0_rapt, file)
            file.close()

            # mspec
            mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
            min_level = np.exp(-100 / 20 * np.log(10))
            stft = pySTFT(wav).T
            stft_mel = np.dot(stft, mel_basis)
            stft_db = 20 * np.log10(np.maximum(min_level, stft_mel)) - 16
            mspec = (stft_db + 100) / 100        
            # plot_f0_mel(f0_rapt, mspec, fs)
            # input()
            # save mspec
            mspec_dir_path = os.path.join(
                root_dir,
                spk_dir,
                'mspec'
            )
            if not os.path.exists(mspec_dir_path):
                os.makedirs(mspec_dir_path)
            mspec_file_path = os.path.join(
                mspec_dir_path,
                spk_audio_file[:-4] + '.pkl'

            )
            file = open(mspec_file_path, 'wb')
            pickle.dump(mspec, file)
            file.close()



def get_hprc_data(filename, root_dir):
    """
    Load all the data with this filename and store it in
    a dict and return it.
    """
    assert os.path.exists(root_dir)
    all_data = {
        'filename': filename,
        'audio': None,
        'text': None,
        'phonemes': None,
        'ema': None,
        'tvs': None,
        'tvs_norm': None,
        'f0': None,
        'mspec': None,
    }
    spk = filename.split('_')[0]
    spk_dir = os.path.join(root_dir, spk)
    _, data_dirs, _ = next(os.walk(spk_dir))

    # audio
    audio_dir = os.path.join(spk_dir, 'audio')
    audio_files = os.listdir(audio_dir)
    audio_path = next((f for f in audio_files if filename in f), None)
    assert audio_path is not None, 'audio for given filename not found.'
    wav, fs = sf.read(os.path.join(audio_dir, audio_path))
    all_data['audio'] = wav

    # text
    text_dir = os.path.join(spk_dir, 'text')
    text_files = os.listdir(text_dir)
    text_path = next((f for f in text_files if filename in f), None)
    assert text_path is not None, 'text for given filename not found.'
    with open(os.path.join(text_dir, text_path), 'r') as tf:
        text = tf.read()
    all_data['text'] = text

    # phonemes
    phon_dir = os.path.join(spk_dir, 'phonemes')
    phon_files = os.listdir(phon_dir)
    phon_path = next((f for f in phon_files if filename in f), None)
    assert phon_path is not None, 'phoneme for given filename not found.'
    grid = textgrids.TextGrid(os.path.join(phon_dir, phon_path))
    all_data['phonemes'] = grid

    # ema
    ema_dir = os.path.join(spk_dir, 'ema')
    ema_files = os.listdir(ema_dir)
    ema_path = next((f for f in ema_files if filename in f), None)
    assert ema_path is not None, 'ema for given filename not found.'
    ema_file = open(os.path.join(ema_dir, ema_path), 'rb')
    ema_data = pickle.load(ema_file)
    ema_file.close()
    all_data['ema'] = ema_data

    # tvs
    tvs_dir = os.path.join(spk_dir, 'tvs')
    tvs_files = os.listdir(tvs_dir)
    tvs_path = next((f for f in tvs_files if filename in f), None)
    assert tvs_path is not None, 'tvs for given filename not found.'
    tvs_file = open(os.path.join(tvs_dir, tvs_path), 'rb')
    tvs_data = pickle.load(tvs_file)
    tvs_file.close()
    all_data['tvs'] = tvs_data

    # tvs norm
    tvs_norm_dir = os.path.join(spk_dir, 'tvs_norm')
    tvs_norm_files = os.listdir(tvs_norm_dir)
    tvs_norm_path = next((f for f in tvs_norm_files if filename in f), None)
    assert tvs_norm_path is not None, 'tvs norm for given filename not found.'
    tvs_norm_file = open(os.path.join(tvs_norm_dir, tvs_norm_path), 'rb')
    tvs_norm_data = pickle.load(tvs_norm_file)
    tvs_norm_file.close()
    all_data['tvs_norm'] = tvs_norm_data

    # f0
    f0_dir = os.path.join(spk_dir, 'f0')
    f0_files = os.listdir(f0_dir)
    f0_path = next((f for f in f0_files if filename in f), None)
    assert f0_path is not None, 'f0 for given filename not found.'
    f0_file = open(os.path.join(f0_dir, f0_path), 'rb')
    f0_data = pickle.load(f0_file)
    f0_file.close()
    all_data['f0'] = f0_data

    # mspec
    mspec_dir = os.path.join(spk_dir, 'mspec')
    mspec_files = os.listdir(mspec_dir)
    mspec_path = next((f for f in mspec_files if filename in f), None)
    assert mspec_path is not None, 'mspec for given filename not found.'
    mspec_file = open(os.path.join(mspec_dir, mspec_path), 'rb')
    mspec_data = pickle.load(mspec_file)
    mspec_file.close()
    all_data['mspec'] = mspec_data

    return all_data



def plot_rand_hprc_tv_phon(
        hprc_pre_path,
        speaker='RAND',
        rate='N',
        norm=False,
        file_name=None
    ):
    """
    Plots the TV's as well as the phonemes and durations
    for a randomly selected file from the given speaker.
    file_name is the index 1 and 2 if split by '_'. 
    """
    assert os.path.exists(hprc_pre_path)
    assert rate in ['N', 'F']
    valid_speakers = ['F01', 'F02', 'F03', 'F04', 
                       'M01', 'M02', 'M03', 'M04', 'RAND']
    if speaker == 'RAND':
        speaker = random.choice(valid_speakers)
    assert speaker in valid_speakers

    spk_audio_dir = os.path.join(hprc_pre_path, speaker, 'audio')
    _, _, audio_files = next(os.walk(spk_audio_dir))
    files = []
    for audio_file in audio_files:
        if audio_file[:-4].split('_')[-1] == rate:
            if file_name:
                if file_name in audio_file:
                    files.append(audio_file[:-4])
            else:
                files.append(audio_file[:-4])
    file = random.choice(files)
    data = get_hprc_data(file, hprc_pre_path)

    # get data to plot
    text = data['text']

    pl, ts = decode_textgrid(data['phonemes'])
    ts = convert_ts_float(str(ts))
    tvs = data['tvs_norm'] if norm else data['tvs']
    tvs_maxy = round(max([max(tv[1]) for tv in tvs.items()]), 2)
    time_steps = len(tvs['LA'])
    time = np.arange(0, time_steps)

    ts_pos = []
    pl_pos = []
    for i, pos in enumerate(ts):
        # last one is different
        if i == len(ts) - 1:
            continue
        else:
            ts_pos.append(pos[1])
    for pos in ts:
        middle = round(pos[0] + ((pos[1] - pos[0]) / 2), 2)
        pl_pos.append(middle)

    # plot data
    fig = plt.figure(figsize=(12,8))
    plt.xlabel('Time [s]')
    plt.xticks([round(pos * 100, 2) for pos in ts_pos], ts_pos,
               color='silver', fontsize=10, rotation=90, alpha=1)
    plt.xlim(0, len(time))

    plt.plot(time, tvs['LA'], label='LA', linestyle='dotted', color='orange')
    plt.plot(time, tvs['LP'], label='LP', color='gold')
    plt.plot(time, tvs['JA'], label='JA', linestyle='dotted', color='gray')
    plt.plot(time, tvs['TTCL'], label='TTCL', color='green')
    plt.plot(time, tvs['TTCD'], label='TTCD', linestyle='dotted', color='green')
    plt.plot(time, tvs['TMCL'], label='TMCL', color='mediumvioletred')
    plt.plot(time, tvs['TMCD'], label='TMCD', linestyle='dotted', color='mediumvioletred')
    plt.plot(time, tvs['TBCL'], label='TBCL', color='steelblue')
    plt.plot(time, tvs['TBCD'], label='TBCD', linestyle='dotted', color='steelblue')

    # phoneme labels
    for i, pos in enumerate(pl_pos):
        # length
        pl_l = 2 if pl[i] == '(...)' else len(pl[i])
        plt.text(pos*100-pl_l, tvs_maxy*0.999, pl[i], color='silver', alpha=1)

    # phoneme durations
    for pos in ts_pos:
        plt.axvline(x=pos*100, color='silver', linestyle='dashed', alpha=0.75)

    fig.legend(
        loc='outside lower center',
        ncols=9,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )
    plt.title(f'"{text}" ({file})')
    plt.show()



def wav2vec2_pr_output(model, wav_path, model_dir_path):
    """
    Extracts the w2v2 based phoneme embedding for the given wav.
    returns phoneme logits and hidden_states of last transformer layer.
    """
    assert os.path.exists(wav_path)
    assert os.path.exists(model_dir_path)
    with open(os.path.join(model_dir_path, 'vocab.json')) as jf:
        vocab = json.load(jf)
    best_chkp_path = os.path.join(model_dir_path, 'best-model-ckpt')
    assert os.path.exists(best_chkp_path)
    file_name = wav_path.split('/')[-1][:-4]
    speaker_dir_path = '/'.join(wav_path.split('/')[:-2])
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    # w2v2PR_model = modules.Wav2Vec2_PR.from_pretrained(best_chkp_path).to(t.device(device))

    wav, fs = librosa.load(wav_path, sr=16000)
    wav_t = t.from_numpy(wav).to(t.device(device))
    # model_out = w2v2PR_model.predict(wav_t, vocab)
    model_out = model.predict(wav_t, vocab)

    return model_out['hidden_states'], model_out['phoneme_logits']


def hprc_lowpass_ema(hprc_pre_dir, cut_freq=10, fs=100, order=5):
    """
    fs: HRPC EMA data is recorded at 100Hz
    order: 5 (like other papers)

    Ignoring the palate data.
    
    HPRC_prep -> Ema dir
    
    new dir: ema_lowpass -> give other methods a lowpass flag
        (if Ture, use ema_lowpass, False: ema)
    
    """
    _, spk_dirs, _ = next(os.walk(hprc_pre_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        _, sub_dirs, _ = next(os.walk(os.path.join(hprc_pre_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'ema':
                spk_ema_dir = os.path.join(hprc_pre_dir, spk_dir, sub_dir)
                _, _, ema_files = next(os.walk(spk_ema_dir))
                for ema_file in ema_files:
                    if '.pkl' in ema_file:
                        print(ema_file)
                        # load ema data
                        ema_path = os.path.join(spk_ema_dir, ema_file)
                        file = open(ema_path, 'rb')
                        ema_data = pickle.load(file)
                        file.close()

                        # TODO: check for nan values and interpolate them from neighborhood

                        if 'palate' in ema_file:
                            # palate.pkl only has dict with x,y,z
                            ema_low = dict()
                            for axis, vals in ema_data.items():
                                if any(np.isnan(vals)):
                                    vals = interpolate_nan(vals)
                                ys = butter_lowpass_filter(
                                    vals, cut_freq, fs, order
                                )
                                ema_low[axis] = ys

                            # Plot example (only z relevant for TV's)
                            # plt.plot(ema_data['z'], color='red', alpha=0.5)
                            # plt.plot(ema_low['z'], color='green', alpha=0.5)
                            # plt.show()
                        else:
                            # lowpass filter ema data 
                            ema_low = dict()
                            for ema_key, axis_dict in ema_data.items():
                                ema_low[ema_key] = dict()
                                for axis, vals in axis_dict.items():
                                    # interpolate nan values
                                    if any(np.isnan(vals)):
                                        vals = interpolate_nan(vals)
                                    # lowpass
                                    ys = butter_lowpass_filter(
                                        vals, cut_freq, fs, order
                                    )
                                    ema_low[ema_key][axis] = ys
                            # Plot example
                            # plt.plot(ema_data['TR']['x'], color='red', alpha=0.5)
                            # plt.plot(ema_low['TR']['x'], color='green', alpha=0.5)
                            # plt.show()
                            # input()

                        # store it in new ema_lowpass dir
                        ema_low_dir_path = os.path.join(
                            hprc_pre_dir,
                            spk_dir,
                            'ema_low'
                        )
                        if not os.path.exists(ema_low_dir_path):
                            os.makedirs(ema_low_dir_path)
                        ema_low_file_path = os.path.join(ema_low_dir_path, ema_file)
                        file = open(ema_low_file_path, 'wb')
                        pickle.dump(ema_low, file)
                        file.close()



def interpolate_signal(org_sig, tar_len):
    org_time_ax = np.arange(len(org_sig))
    tar_time_ax = np.linspace(0, len(org_sig) - 1, tar_len)
    
    interpolator = scipy.interpolate.interp1d(org_time_ax, org_sig, kind='linear', axis=0)
    interpolated_sig = interpolator(tar_time_ax)
    return interpolated_sig

    
def interpolate_TVs_49hz(hprc_pre_dir, model_dir_path):
    """
    Resamples all TVs and TVs_norm to 49hz and stores them in
    respective new sub_dir's: TVs_49hz and TVs_norm_49hz.

    This is required for the RNN baseline, since wav2vec2.0 has an
    output frequency of 49 samples per second, while the EMA data
    (and thus the resulting TV's) are sampled at 100Hz.
    """
    assert os.path.exists(hprc_pre_dir)
    assert os.path.exists(model_dir_path)

    # load model
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    best_ckpt_path = os.path.join(model_dir_path, 'best-model-ckpt')
    model_cfg = pickle.load(open(os.path.join(best_ckpt_path, 'model_cfg.pkl'), 'rb'))
    w2v2PR_model = Wav2Vec2_PR(
        model_cfg['pretrain_cfg'],
        model_cfg['cache_dir'],
        model_cfg['huggingface_model_id']
    ).to(device)
    best_ckpt = t.load(
        os.path.join(best_ckpt_path, 'pytorch_model.bin'),
        map_location=t.device(device),
    )
    w2v2PR_model.load_state_dict(best_ckpt)

    # loop for TVs
    _, spk_dirs, _ = next(os.walk(hprc_pre_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_tvs_49hz_dir = os.path.join(hprc_pre_dir, spk_dir, 'tvs_49hz')
        _, sub_dirs, _ = next(os.walk(os.path.join(hprc_pre_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs':
                hprc_spk_tvs_dir = os.path.join(hprc_pre_dir, spk_dir, sub_dir)
                _, _, tv_files = next(os.walk(hprc_spk_tvs_dir))
                for tv_file in tv_files:
                    if '.pkl' in tv_file and 'palate' not in tv_file:
                        print('\t', tv_file)
                        # file stuff
                        file_name = tv_file[:-4]
                        wav_path = os.path.join(hprc_pre_dir, spk_dir, 'audio', file_name + '.wav')
                        tvs_path = os.path.join(hprc_spk_tvs_dir, tv_file)
                        file = open(tvs_path, 'rb')
                        tvs = pickle.load(file)
                        file.close()

                        # required for resample target length
                        w2v2_hidden, _ = wav2vec2_pr_output(w2v2PR_model, wav_path, model_dir_path)

                        # iterate over tv's and resample each of them to target length
                        tvs_inter = dict()
                        for k, v in tvs.items():
                            v_inter = interpolate_signal(v, w2v2_hidden.shape[0])
                            tvs_inter[k] = v_inter
                        
                        # save in extra dir
                        if not os.path.exists(spk_tvs_49hz_dir):
                            os.makedirs(spk_tvs_49hz_dir)
                        spk_tvs_49hz_file = os.path.join(
                            spk_tvs_49hz_dir,
                            tv_file
                        )
                        file = open(spk_tvs_49hz_file, 'wb')
                        pickle.dump(tvs_inter, file)
                        file.close()

    # loop for TVs_norm
    _, spk_dirs, _ = next(os.walk(hprc_pre_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_tvs_norm_49hz_dir = os.path.join(hprc_pre_dir, spk_dir, 'tvs_norm_49hz')
        _, sub_dirs, _ = next(os.walk(os.path.join(hprc_pre_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs_norm':
                hprc_spk_tvs_norm_dir = os.path.join(hprc_pre_dir, spk_dir, sub_dir)
                _, _, tv_norm_files = next(os.walk(hprc_spk_tvs_norm_dir))
                for tv_norm_file in tv_norm_files:
                    if '.pkl' in tv_norm_file and 'palate' not in tv_norm_file:
                        print('\t', tv_norm_file)
                        # file stuff
                        file_name = tv_norm_file[:-4]
                        wav_path = os.path.join(hprc_pre_dir, spk_dir, 'audio', file_name + '.wav')
                        tvs_norm_path = os.path.join(hprc_spk_tvs_norm_dir, tv_norm_file)
                        file = open(tvs_norm_path, 'rb')
                        tvs_norm = pickle.load(file)
                        file.close()

                        # required for resample target length
                        w2v2_hidden, _ = wav2vec2_pr_output(w2v2PR_model, wav_path, model_dir_path)

                        # iterate over tv's and resample each of them to target length
                        tvs_norm_inter = dict()
                        for k, v in tvs_norm.items():
                            v_inter = interpolate_signal(v, w2v2_hidden.shape[0])
                            tvs_norm_inter[k] = v_inter
                        
                        # save in extra dir
                        if not os.path.exists(spk_tvs_norm_49hz_dir):
                            os.makedirs(spk_tvs_norm_49hz_dir)
                        spk_tvs_norm_49hz_file = os.path.join(
                            spk_tvs_norm_49hz_dir,
                            tv_norm_file
                        )
                        file = open(spk_tvs_norm_49hz_file, 'wb')
                        pickle.dump(tvs_norm_inter, file)
                        file.close()




def hprc_csv_phn_frames_49hz(hprc_pre_dir, model_dir_path):
    """
    # 45 or so per time-step (one-hot) -> or do this rather in the model
    Per T (frame) of wav2vec2 output frequency, provide the vocab based
    ground truth phoneme label (based on MAUS force alignment).
    This has to then be one-hot encoded at a later point for CE loss.

    Adjusts the hprc.csv: for each row in there, compute the phn_frames_49 
    (based on the phonemes and phonemes_timestamps, as well as T from 
    wav2vec2) and add it to the csv row.
    """
    assert os.path.exists(hprc_pre_dir)
    assert os.path.exists(model_dir_path)
    hprc_csv_path = os.path.join(hprc_pre_dir, 'hprc.csv')
    assert os.path.exists(hprc_csv_path)
    with open(os.path.join(model_dir_path, 'vocab.json'), 'r') as f:
        vocab = json.load(f)
    vocab.pop('(blank)')
    
    # load model
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    best_ckpt_path = os.path.join(model_dir_path, 'best-model-ckpt')
    model_cfg = pickle.load(open(os.path.join(best_ckpt_path, 'model_cfg.pkl'), 'rb'))
    w2v2PR_model = Wav2Vec2_PR(
        model_cfg['pretrain_cfg'],
        model_cfg['cache_dir'],
        model_cfg['huggingface_model_id']
    ).to(device)
    best_ckpt = t.load(
        os.path.join(best_ckpt_path, 'pytorch_model.bin'),
        map_location=t.device(device),
    )
    w2v2PR_model.load_state_dict(best_ckpt)
    
    # load csv file into dataframe
    csv_df = pd.read_csv(hprc_csv_path)
    # new column name
    csv_df['phn_frames_49hz'] = None

    for idx, row in csv_df.iterrows():
        print(row)
        phn_tokens = phonemes_idx(vocab, row.phoneme_labels)
        w2v2_hid, _ = wav2vec2_pr_output(w2v2PR_model, row.path_wav, model_dir_path)
        w2v2_T = w2v2_hid.shape[0]

        ts_floats = ast.literal_eval(row.phoneme_timestamps)
        ts_floats[-1] = round(ts_floats[-1], 2)

        # discretize phonemes into 20ms frames
        phn_frames_49hz = match_phonemes_to_frames(ts_floats, phn_tokens, 0.02)
        # correct for length discrepancies
        diff = abs(len(phn_frames_49hz) - w2v2_T)
        phn_frames_49hz = phn_frames_49hz[:-diff]
                
        assert len(phn_frames_49hz) == w2v2_T 
        # add new discreticed phoneme frames to df 
        row['phn_frames_49hz'] = phn_frames_49hz
        csv_df.iloc[idx] = row

    # save updated df to hprc.csv
    csv_df.to_csv(hprc_csv_path, index=False)

    
                        
def min_max_spk_tv_hprc(hprc_prep_csv_path, tv, rate, speaker):
    """
    Returns the max and minium value for a given
    speaker, rate, and tv.
    Base on the ORIGINAL (unormalized, not resampled) TV data.
    """
    assert os.path.exists(hprc_prep_csv_path)
    assert rate in ['F', 'N', 'both']
    assert speaker in ['F01', 'F02', 'F03', 'F04', 
                       'M01', 'M02', 'M03', 'M04']
    assert tv in ['LA', 'LP', 'JA', 'TTCL', 'TTCD',
                  'TMCL', 'TMCD', 'TBCL', 'TBCD']
    
    hprc_df = pd.read_csv(hprc_prep_csv_path)
    hprc_spk = hprc_df[hprc_df.speaker == speaker]
    if rate == 'both':
        hprc_spk_rate = hprc_spk
    else:
        hprc_spk_rate = hprc_spk[hprc_spk.rate == rate]

    all_tv_values = []
    for idx, row in hprc_spk_rate.iterrows():
        with open(row.path_tvs, 'rb') as file:
            tmp_tv = pickle.load(file)
        all_tv_values.extend(tmp_tv[tv])

    return min(all_tv_values), max(all_tv_values)


def spk_onehot_emb(tgt_spk):
    hprc_spks = ['M01', 'M02', 'M03', 'M04',
                 'F01', 'F02', 'F03', 'F04']
    spk_to_idx = {spk: i for i, spk in enumerate(hprc_spks)}
    idx = spk_to_idx[tgt_spk]
    one_hot = t.eye(len(hprc_spks))[idx]
    return one_hot







def last_step_tv_smoothing(hprc_pre_dir, gauss_k=2):
    """
    
    """
    assert os.path.exists(hprc_pre_dir)
    
    # loop for TVs_norm
    _, spk_dirs, _ = next(os.walk(hprc_pre_dir))
    for spk_dir in spk_dirs:
        print(spk_dir)
        spk_tvs_norm_49hz_dir = os.path.join(hprc_pre_dir, spk_dir, 'tvs_norm_49hz_gaus')
        _, sub_dirs, _ = next(os.walk(os.path.join(hprc_pre_dir, spk_dir)))
        for sub_dir in sub_dirs:
            if sub_dir == 'tvs_norm_49hz':
                hprc_spk_tvs_norm_49hz_dir = os.path.join(hprc_pre_dir, spk_dir, sub_dir)
                _, _, tv_norm_49hz_files = next(os.walk(hprc_spk_tvs_norm_49hz_dir))
                for tv_norm_49hz_file in tv_norm_49hz_files:
                    if '.pkl' in tv_norm_49hz_file and 'palate' not in tv_norm_49hz_file:
                        print('\t', tv_norm_49hz_file)
                        # load tv data
                        tvs_norm_49hz_path = os.path.join(hprc_spk_tvs_norm_49hz_dir, tv_norm_49hz_file)
                        file = open(tvs_norm_49hz_path, 'rb')
                        tvs_data = pickle.load(file)
                        file.close()

                        tv = 'JA'
                        tv_data_gauss_1 = gaussian_filter1d(tvs_data[tv], 1)
                        tv_data_gauss_2 = gaussian_filter1d(tvs_data[tv], 2)
                        tv_data_gauss_3 = gaussian_filter1d(tvs_data[tv], 3)

                        plt.plot(tvs_data[tv], color='black', alpha=0.5)
                        plt.plot(tv_data_gauss_1, color='blue', alpha=0.5)
                        plt.plot(tv_data_gauss_2, color='red', alpha=0.5)
                        plt.plot(tv_data_gauss_3, color='green', alpha=0.5)
                        plt.show()


                        input()







if __name__ == '__main__':
    # 1) extract audio, txt, and ema data
    # hprc_path = '../data/HPRC'
    # hprc_processing(hprc_path, resample_fs=16000)

    # 2) based on extracted data, create more data
    hprc_pre_path = '../data/HPRC_prep'
    model_dir_path = '../models/w2v2_phon_rec/wav2vec2-xlsr-53'
    
    # 2.1 - F0, mspec, mfccs
    # hprc_f0(hprc_pre_path) # -> TODO
    # hprc_mspec(hprc_pre_path)
    # hprc_mspec_znorm(hprc_pre_path)
    # hprc_f0_mspec(hprc_pre_path) OLD (combinding both)
    # hprc_mfccs(hprc_pre_path)
    
    # 2.2 - MAUS phoneme labels
    # hprc_phoneme(hprc_pre_path, replace=False)

    # 2.3 lowpass filter the EMA data (hprc_prep -> spk -> ema_low
    # TODO if not calling lowpass - also fix nan problem for ema !!!
    # hprc_lowpass_ema(hprc_pre_path) 
    
    # 2.4 - TVs
    # hprc_tvs(hprc_pre_path, lowpass=True)
    
    # 2.5 - normalize (select one)
    # tvs_minmax_utterance(hprc_pre_path, 'N') # -> TODO
    # tvs_minmax_speaker(hprc_pre_path, 'both')
    # tvs_minmax_global(hprc_pre_path, 'N') # -> TODO
    # tvs_zscore_utterance(hprc_pre_path, 'both') # -> best
    # tvs_zscore_speaker(hprc_pre_path, 'N') # TODO check if this is correct -> DONT THINK SO
    # tvs_zscore_global(hprc_pre_path, 'both') # -> TODO

    # 3) interpolate TVs (normal and normalized) to 49Hz
    # interpolate_TVs_49hz(hprc_pre_path, model_dir_path)

    # test) gaussian smooth the TV's as a last step?
    # last_step_tv_smoothing(hprc_pre_path) # TODO not finished -> not needed i guess

    # 4) create hprc.csv 
    # hprc_csv(hprc_pre_path)
    hprc_csv_phn_frames_49hz(hprc_pre_path, model_dir_path)









    
    # 98) plotting
    # NOTE: won't work, since changed phoneme_timestamps from list of
    # tuples to list of single values (len is one longer than phonemes),
    # since middle value is used twice
    # plot_rand_hprc_tv_phon(
    #     hprc_pre_path=hprc_pre_path,
    #     speaker='RAND',
    #     rate='N',
    #     norm=True,
    #     file_name=None
    # )

    # 99) get all of the data in a single dict based on a single filename
    # filename = 'F01_B01_S01_R01_N'
    # f_data = get_hprc_data(filename, hprc_pre_path)

    print('\n-> main done.')


