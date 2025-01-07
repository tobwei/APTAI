"""
Contains functions/classes around the CommonPhone dataset.
Important contents:
    - Pytorch Dataset
    - function to create commonphone.csv
"""
import torch as t
import torchaudio
import random
from utility import (
    phonemes_idx,
    convert_ts_float
)



class CommonPhoneDataset(t.utils.data.Dataset):
    """
    No Tokenizer, only vocab, re-sampling to match Wav2Vec2.0 pre-training audio.
    could be used : spk_label = row.speaker for speaker_head
    """
    def __init__(self, df, vocab, cropping=False):
        self.df = df
        self.vocab = vocab
        self.cropping = cropping

    def __getitem__(self, index):
        # load audio + resample (wav2vec2 was pretrained with 16khz audio)
        row = self.df.iloc[index]
        audio, fs = torchaudio.load(row.path)
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=fs, new_freq=16_000,
        )[0]

        if self.cropping:
            # audio (1 second chunk) - no padding since: min_wav_dur of CP_eng is 1.176
            duration_samples = 16000 # fs=16000
            start_sample, last_sample = 0, len(audio -1)
            rand_start_sample = random.randint(start_sample, last_sample - duration_samples)
            new_end_sample = rand_start_sample + duration_samples
            indices = t.LongTensor(range(rand_start_sample, new_end_sample))
            audio = t.index_select(audio, 0, indices)

            # length
            audio_len = len(audio)

            # label
            # rand_start_seconds = round(rand_start_sample / 16000, 2)
            # new_end_seconds = round(new_end_sample / 16000, 2)
            rand_start_seconds = rand_start_sample / 16000
            new_end_seconds = new_end_sample / 16000
            phoneme_ts = row.phoneme_timestamps
            phoneme_ts_tupels = convert_ts_float(phoneme_ts)

            phoneme_labels_in_crop = []
            for phoneme_ts_tupel in phoneme_ts_tupels:
                # first tupel index
                if phoneme_ts_tupel[0] <= rand_start_seconds < phoneme_ts_tupel[1]:
                    phoneme_labels_in_crop.append(phoneme_ts_tupels.index(phoneme_ts_tupel))
                # last tupel index
                if phoneme_ts_tupel[0] < new_end_seconds <= phoneme_ts_tupel[1]:
                    phoneme_labels_in_crop.append(phoneme_ts_tupels.index(phoneme_ts_tupel))

            assert len(phoneme_labels_in_crop) == 2
            phoneme_labels_in_crop = [i for i in range(
                phoneme_labels_in_crop[0],
                phoneme_labels_in_crop[1] + 1)]
            phoneme_labels = row.phonemes
            phoneme_label = [phoneme_labels.split(' ')[i] for i in phoneme_labels_in_crop]
            phoneme_label = (' ').join(phoneme_label)
            phoneme_label = phonemes_idx(self.vocab, phoneme_label)

        else:
            # use entire wav
            audio_len = len(audio)
            phoneme_label = phonemes_idx(self.vocab, row.phonemes)

        return {
            # Input for forward
            'audio': audio,
            'audio_len': audio_len,
            'phoneme_label' : phoneme_label,
        }

    def __len__(self):
        return len(self.df)



def commonphone_csv(cp_path, langs=['en']):
    """
    Create a commonphone.csv file which contains:
        index,lang,path,speaker,text,phonemes,split
    for all languages that are given in the parameter of the CommonPhone dataset.

    Args:
        languages: list of languages that should be included form the CommonPhone dataset.
            possible values: de, en, es, fr, it, ru
    """
    val_langs = ['de', 'en', 'es', 'fr', 'it', 'ru']
    assert os.path.exists(cp_path)
    ds_path = cp_path
    csv_path = Path('/'.join(ds_path.split('/')[:-1])) / 'commonphone.csv'

    assert set(langs).intersection(set(val_langs)) == set(langs), \
        'Please only enter valid languages: de, en, es, fr, it, ru as a list of strings.'

    data = []
    index = 0
    _, lang_dirs, _ = next(os.walk(ds_path))
    
    for lang_dir in lang_dirs:
        if lang_dir in langs:
            train_csv = Path(os.path.join(ds_path, lang_dir, 'train.csv'))
            val_csv = Path(os.path.join(ds_path, lang_dir, 'dev.csv'))
            test_csv = Path(os.path.join(ds_path, lang_dir, 'test.csv'))

            # train
            train_df = pd.read_csv(train_csv)
            for _, df_row in train_df.iterrows():
                row = []
                wav = df_row['audio file'].split('.')[0] + '.wav'
                path = Path(os.path.join(ds_path, lang_dir, 'wav', wav))
                lang = lang_dir
                speaker = df_row['id']
                split = 'train'
                phoneme_list, text_list, phoneme_timestamps = get_CommonPhonewav_labels(path, ds_path)
                phoneme_labels = ''
                for p in phoneme_list:
                    phoneme_labels += p + ' '
                phoneme_labels = phoneme_labels[:-1] # get rid of space
                text_labels = ''
                for w in text_list:
                    if w == '':
                        continue
                    text_labels += w + ' '
                
                row.append(index)
                row.append(lang)
                row.append(path)
                row.append(speaker)
                row.append(text_labels)
                row.append(phoneme_labels)
                row.append(phoneme_timestamps)
                row.append(split)

                data.append(row)
                index += 1

            # validation
            val_df = pd.read_csv(val_csv)
            for _, df_row in val_df.iterrows():
                row = []
                wav = df_row['audio file'].split('.')[0] + '.wav'
                path = Path(os.path.join(ds_path, lang_dir, 'wav', wav))
                speaker = df_row['id']
                split = 'val'
                phoneme_list, text_list, phoneme_timestamps = get_CommonPhone_wav_labels(path, ds_path)
                phoneme_labels = ''
                for p in phoneme_list:
                    phoneme_labels += p + ' '
                phoneme_labels = phoneme_labels[:-1] # get rid of space
                text_labels = ''
                for w in text_list:
                    if w == '':
                        continue
                    text_labels += w + ' '
                
                row.append(index)
                row.append(lang)
                row.append(path)
                row.append(speaker)
                row.append(text_labels)
                row.append(phoneme_labels)
                row.append(phoneme_timestamps)
                row.append(split)

                data.append(row)
                index += 1

            # test
            test_df = pd.read_csv(test_csv)
            for _, df_row in test_df.iterrows():
                row = []
                wav = df_row['audio file'].split('.')[0] + '.wav'
                path = Path(os.path.join(ds_path, lang_dir, 'wav', wav))
                speaker = df_row['id']
                split = 'test'
                phoneme_list, text_list, phoneme_timestamps = get_CommonPhone_wav_labels(path, ds_path)
                phoneme_labels = ''
                for p in phoneme_list:
                    phoneme_labels += p + ' '
                phoneme_labels = phoneme_labels[:-1] # get rid of space
                text_labels = ''
                for w in text_list:
                    if w == '':
                        continue
                    text_labels += w + ' '
                
                row.append(index)
                row.append(lang)
                row.append(path)
                row.append(speaker)
                row.append(text_labels)
                row.append(phoneme_labels)
                row.append(phoneme_timestamps)
                row.append(split)

                data.append(row)
                index += 1
        
    # save as commonphone.csv file
    columns = ['index', 'lang', 'path', 'speaker', 'text', 'phonemes', 'phoneme_timestamps', 'split']
    with open(csv_path, 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerow(columns)
        write.writerows(data)



def trim_CommonPhone_csv(csv_path, num_train=32, num_val=5, num_test=5):
    """
    Only used for local laptop debugging - strips the commonphone.csv down a lot:
    """
    new_csv_path = os.path.join('/'.join(csv_path.split('/')[:-1]), 'commonphone_trimmed.csv')
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)

    df_train = df[df.split == 'train']
    df_val = df[df.split == 'val']
    df_test = df[df.split == 'test']

    df_train = df_train.sample(num_train)
    df_val = df_val.sample(num_val)
    df_test = df_test.sample(num_test)

    df_new = pd.concat([df_train, df_val, df_test])
    df_new.to_csv(new_csv_path, index=False)



def get_CommonPhone_wav_labels(wav_path, cp_path):
    """
    Returns the phoneme and text labels as lists (based on the praat file) for the given wav file of CommonPhone.
    
    Params:
        wav_path: the wav file for which the praat based phoneme lables should be returned.
    """
    assert os.path.exists(wav_path)
    lang = str(wav_path).split('/')[-3]
    fname = str(wav_path).split('/')[-1][:-4]
    ds_path = Path(cp_path)
    assert os.path.exists(ds_path)
    grid_path = ds_path / Path(lang) / 'grids'
    assert os.path.exists(grid_path)
    wav_grid_path = grid_path / Path(fname + '.TextGrid')
    assert os.path.exists(wav_grid_path)
    grid = textgrids.TextGrid(wav_grid_path)

    # get phoneme labels and timestamps in two chronological lists,
    # where for timestamps each list element is a tupel: (start, end)
    phoneme_labels = []
    phoneme_timestamps = []
    for mau in grid['MAU']:
        phoneme_labels.append(mau.text.transcode())
        phoneme_timestamps.append((mau.xmin, mau.xmax))
        # Print label and syllable duration, CSV-like
        # print('"{}";{}'.format(label, syll.dur))

    # get text transcriptions
    text_labels = []
    for ort_mau in grid['ORT-MAU']:
        label = ort_mau.text.transcode()
        text_labels.append(label)

    # print(grid)
    return phoneme_labels, text_labels, phoneme_timestamps



def remap_commonphone_speaker(cp_path):
    """
    Re-maps speaker (originally a large string) to a simple int
    so that it can be used as a label for CE classification.
    """
    assert os.path.exists(cp_path)
    ds_path = cp_path
    csv_path = Path('/'.join(ds_path.split('/')[:-1])) / 'commonphone.csv'

    df = pd.read_csv(csv_path)

    # map unique speaker's to a int label
    map = dict()
    label = 0
    for spk in df['speaker'].unique():
        map[spk] = label
        label += 1

    for i, row in df.iterrows():
        speaker = row['speaker']
        label = map[speaker]
        df.at[i,'speaker'] = label

    df.to_csv(csv_path, index=False)












if __name__ == '__main__':
    # 1) create csv and remap speaker id's 
    # cp_path = '../data/CommonPhone/CP'
    # commonphone_csv(cp_path, langs=['en'])
    # remap_commonphone_speaker(cp_path)
    # 1.1) create a trimmed version of the csv for debugging
    # csv_path = '../data/CommonPhone/commonphone.csv'
    # trim_cp_csv(csv_path, num_train=8, num_val=4, num_test=4)

    # 2) plotting stuff
    # print(f'Min duration in CommonPhone (eng) is {min_audio_duration(csv_path)}.')
    # p, t, ts = get_CommonPhone_wav_labels(
    #     '../data/CommonPhone/CP/en/wav/common_voice_en_292.wav',
    #     cp_path_lap
    # )

    print('\n-> main done.')
