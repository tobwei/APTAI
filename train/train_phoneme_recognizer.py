"""
https://huggingface.co/learn/audio-course/chapter3/ctc?fw=pt
"""
import argparse
import pickle
import os
import random
import math
import json
import wandb
import tqdm
import editdistance
import itertools
import torch as t
import torchaudio.models.decoder
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import transformers
from transformers.integrations import is_wandb_available
from transformers import (
    logging,
)
logging.set_verbosity_warning()
logging.set_verbosity_error()
    
from modules import (
    Wav2Vec2_PR,
)
from utility import (
    init_logger,
    count_parameters,
    print_children,
    idx_phonemes,
    _ctc_decode
)
from dataset_commonphone import (
    CommonPhoneDataset,
)
from dataset_hprc import(
    HPRCDataset,
)



# ################################
# Arguments
# ################################
def parse_args():
    parser = argparse.ArgumentParser(
        description = ('Wav2Vec2.0 (XLSR-53) based Phoneme Recognizer.')
    )

    # Dirs, Paths, Misc
    parser.add_argument(
        '--exp_dir', type=Path, default=None,
        help='If provided continue training, otherwise start new tarining run.'
    )
    parser.add_argument(
        '--cache_dir', type=Path, default=Path('../.cache'),
        help='Where do you want to store the pretrained models downloaded from huggingface.co?'
    )
    parser.add_argument(
        '--logging', action=argparse.BooleanOptionalAction,
        help='If True, enable wandb logging, if false disable it (mostly for debugging).'
    )
    parser.add_argument(
        '--laptop', action=argparse.BooleanOptionalAction,
        help='Only a couple of files in train/val/test.'
    )
    parser.add_argument(
        '--prefix', type=str, default='',
        help='Custom string to add at the beginning of the experiment name.'
    )
    parser.add_argument(
        '--cp_csv_path', type=Path, default=Path('../data/CommonPhone/commonphone.csv'),
        help='Contains the CommonPhone dataset to use, including splits.'
    )
    parser.add_argument(
        '--hprc_csv_path', type=Path, default=Path('../data/HPRC_prep/hprc.csv'),
        help='Contains the hrpc dataset for testing.'
    )
    parser.add_argument(
        '--cropping', action=argparse.BooleanOptionalAction,
        help='If False: use entire wavs, if True: use random 1second crops per wav file (training only).'
    )

    # Training
    parser.add_argument(
        '--pretrained_weights', type=Path, default=None,
        help='If provided, continue model training.'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=160,
        help='Number of training epochs to perform to step through the training data.'
    )
    parser.add_argument(
        '--num_warmup_epochs', type=int, default=0,
        help='Number of epochs for warming up the learning rate.'
    )
    parser.add_argument(
        '--num_static_epochs', type=int, default=0,
        help='Number of epochs with static learning rate (following warmup_epochs).'
    )
    parser.add_argument(
        '--samples_per_epoch', type=int, default=2000,
        help='Number of samples per epoch. Needs to be divisible by batch_size and <= len(dataset).'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for training.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=5e-4,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--lr_decay', type=float, default=0.96,
        help='The amount of decay to apply to the lr after warmup and static epochs.'
    )
    parser.add_argument(
        '--adam_beta1', type=float, default=0.9,
        help='Beta1 for Adam optimizer.'
    )
    parser.add_argument(
        '--adam_beta2', type=float, default=0.999,
        help='Beta2 for Adam optimizer.'
    )
    parser.add_argument(
        '--adam_epsilon', type=float, default=1e-8,
        help='Epsilon for Adam optimizer.'
    )
    parser.add_argument(
        '--adam_weight_decay', type=float, default=0.0,
        help='Weight decay parameter for adam.'
    )
    parser.add_argument(
        '--save_all_epochs', type=bool, default=False,
        help='Save weights after each epoch.'
    )
    parser.add_argument(
        '--target_metric', type=str, default="mean_val_per",
        help='Possible: mean_val_per, mean_val_loss'
    )
    parser.add_argument(
        '--target_metric_bigger_better', type=bool, default=False,
        help='For certain metrics, bigger values equal better performance (e.g. val_accuracy)'
    )

    # Model specific
    parser.add_argument(
        '--num_hidden_layers', type=int, default=24,
        help='Number of hidden_layers (i.e. transformer) of the XLSR model to use, max = 24.'
    )
    parser.add_argument(
        '--final_dropout', type=float, default=0.0,
        help='Amount of dropout applied to the output of the final wav2vec2 hidden layer.'
    )
    parser.add_argument(
        '--ten_ms', action=argparse.BooleanOptionalAction,
        help=('If True, changes the conv_stide from [5, 2, 2, 2, 2, 2] to [ ..., 1],' 
            'which leads to 10ms instead of 20ms frame rate for the w2v2 feature encoder.')
    )
    parser.add_argument(
        '--huggingface_model_id', type=str, default='facebook/wav2vec2-large-xlsr-53',
        help='The huggingface pre-trained model that should be fine-tuned for PR.'
    )
    parser.add_argument(
        '--freeze_feature_extractor', type=bool, default=False,
        help='Wav2Vec2 feature extractor frozen or not.'
    )

    # Finalize arugments
    args = parser.parse_args()
    args.date_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
    args.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    args.exp_name = (
        f'{args.prefix}'
        f'_crop{args.cropping}'
        # f'_10ms{args.ten_ms}'
        f'_e{args.num_epochs}'
        f'_bs{args.batch_size}'
        f'_spe{args.samples_per_epoch}'
    )
    if args.laptop:
        args.num_epochs = 1
        args.num_warmup_epochs = 1
        args.num_static_epochs = 1
    
    if args.exp_dir is None:
        # Train from scratch
        args.exp_dir = Path(f'../experiments/phoneme_recognizer/{args.date_time}_{args.exp_name}')
        args.exp_dir.mkdir(parents=True, exist_ok=False)
        args.train_from_ckpt = False
    else:
        # Train from checkpoint
        args.exp_dir = Path(args.exp_dir)
        assert args.exp_dir.exists()
        args.train_from_ckpt = True

    return args


# ################################
# Internal functions
# ################################
def _get_vocab(exp_dir, df):
    """
    Creates a vocab for the current training.
    """
    vocabs = sorted(list(set(itertools.chain.from_iterable(
        df.phonemes.apply(lambda x: x.split())))))
    vocabs[:0] = ['(blank)']
    vocab_dict = {v: i for i, v in enumerate(vocabs)}

    with open(exp_dir / 'vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f)

    return vocab_dict



def _collator(batch):
    return {
        "input_values": t.nn.utils.rnn.pad_sequence(
            [x["audio"] for x in batch],
            batch_first=True,
            padding_value=0.0,
        ),
        "input_lengths": t.LongTensor(
            [x["audio_len"] for x in batch]
        ),
        "phoneme_labels": t.nn.utils.rnn.pad_sequence(
            [t.IntTensor(x["phoneme_label"]) for x in batch],
            batch_first=True,
            padding_value=-100,
        )
    }



def _get_dataset(dataset, vocab, df, batch_size, cropping=False, **kwargs):
    assert dataset in ['cp', 'hprcN', 'hprcF']

    if dataset == 'cp':
        return t.utils.data.DataLoader(
            CommonPhoneDataset(df, vocab, cropping),
            batch_size = batch_size,
            collate_fn=_collator,
            pin_memory=True,
            **kwargs,
        )
    elif dataset == 'hprcN':
        return t.utils.data.DataLoader(
            HPRCDataset(df, vocab, rate='N'),
            batch_size = batch_size,
            collate_fn=_collator,
            pin_memory=True,
            **kwargs,
        )
    elif dataset == 'hprcF':
        return t.utils.data.DataLoader(
            HPRCDataset(df, vocab, rate='F'),
            batch_size = batch_size,
            collate_fn=_collator,
            pin_memory=True,
            **kwargs,
        )



def _get_lr_schedule(
    warmup_epochs: int,
    static_epochs: int,
    lr_decay: float
) -> callable:
    def lambda_lr(epoch):
        """
        Example schedule, with 160 epochs: warmup_epochs = 10, static_epochs = 30, lr_decay=0.96
        First 10 epochs: increase LR by factor 10 (warm-up)
        """
        if epoch < warmup_epochs:
            return 10. * (epoch + 1) / warmup_epochs
        elif epoch < warmup_epochs + static_epochs:
            return 10.
        else:
            return 10. * lr_decay**(epoch-(warmup_epochs + static_epochs))
    return lambda_lr



def _prepare_datasets(exp_dir, batch_size, df, cropping, train_from_ckpt):
    """
    df: is the read .csv file which represents the dataset: here commonphone.csv
    """
    if train_from_ckpt:
        train_df = pd.read_csv(exp_dir / 'train.csv')
        valid_df = pd.read_csv(exp_dir / 'valid.csv')
        test_df = pd.read_csv(exp_dir / 'test.csv')
    else:
        if 'split' in df:
            # Predefined split
            train_df = df[df.split == 'train']
            valid_df = df[df.split == 'val']
            test_df = df[df.split == 'test']
        else:
            # Random splits
            raise NotImplementedError
        
        train_df.to_csv(exp_dir / 'train.csv', index=False)
        valid_df.to_csv(exp_dir / 'valid.csv', index=False)
        test_df.to_csv(exp_dir / 'test.csv', index=False)

    vocab = _get_vocab(exp_dir, df)
    train_dl = _get_dataset('cp', vocab, train_df, batch_size, cropping=cropping, shuffle=True)
    valid_dl = _get_dataset('cp', vocab, valid_df, 1, cropping=False, shuffle=False)
    test_dl = _get_dataset('cp', vocab, test_df, 1, cropping=False, shuffle=False)

    return vocab, train_dl, valid_dl, test_dl



def _prepare_model_optimizer_scheduler(args_cfg, vocab):
    if args_cfg.pretrained_weights is not None:
        # TODO 
        model = Wav2Vec2_PR.from_pretrained(args_cfg.pretrained_weights)
        optimizer = t.optim.Adam(
            model.parameters(),
            lr=args_cfg.learning_rate,
            betas=(args_cfg.adam_beta1, args_cfg.adam_beta2),
            eps=args_cfg.adam_epsilon,
        )
    else:
        # Model & config
        pretrain_cfg, *_ = transformers.PretrainedConfig.get_config_dict(
            args_cfg.huggingface_model_id
        )
        pretrain_cfg['vocab_size'] = len(vocab)
        pretrain_cfg['final_dropout'] = args_cfg.final_dropout
        pretrain_cfg['num_hidden_layers'] = args_cfg.num_hidden_layers
        pretrain_cfg['ctc_loss_reduction'] = 'mean'
        pretrain_cfg['ctc_zero_infinity'] = True
        pretrain_cfg['blank'] = 0 
        if args_cfg.ten_ms:
            pretrain_cfg['conv_stride'] = [5, 2, 2, 2, 2, 2, 1]
        pretrain_cfg = transformers.Wav2Vec2Config.from_dict(pretrain_cfg)
            
        model = Wav2Vec2_PR(
            pretrain_cfg,
            args_cfg.cache_dir,
            args_cfg.huggingface_model_id,
        ).to(args_cfg.device)
        if args_cfg.freeze_feature_extractor:
            model.freeze_feature_encoder()
        
        print('Wav2Vec2_PR config:\n', pretrain_cfg, cfg, '\n')
        print_children(model)

    optimizer = t.optim.Adam(
        model.parameters(),
        lr=args_cfg.learning_rate,
        betas=(args_cfg.adam_beta1, args_cfg.adam_beta2),
        eps=args_cfg.adam_epsilon,
        weight_decay=args_cfg.adam_weight_decay
    )

    lr_scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=_get_lr_schedule(
            args_cfg.num_warmup_epochs,
            args_cfg.num_static_epochs,
            args_cfg.lr_decay,
        )
    )

    return model, optimizer, lr_scheduler



# ################################
# Train, Val, Test Functions
# ################################
def train(
    cfg, model, optimizer, lr_scheduler,
    vocab, train_dataloader, valid_dataloader,
    best_ckpt_path, last_ckpt_path, all_ckpt_path
):
    start_epoch = 0
    eval_target = None

    if cfg.train_from_ckpt:
        lr_scheduler = t.load(last_ckpt_path / 'scheduler.pt')
        # TODO
        model.load_state_dict(Wav2Vec2_PR.from_pretrained(last_ckpt_path).to(t.device('cpu')).state_dict())
        optimizer.load_state_dict(t.load(last_ckpt_path / 'optimizer.pt'))
        start_epoch = lr_scheduler['last_epoch']

    epoch_progress_bar = tqdm.tqdm(iterable=range(cfg.num_epochs), desc='Epochs: ', leave=False)

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_train_steps = int(cfg.samples_per_epoch / cfg.batch_size)
        subset_random_idx = 0
        sum_train_loss = 0

        subset_random = random.sample(range(len(train_dataloader)), epoch_train_steps)
        batch_progress_bar = tqdm.tqdm(iterable=range(epoch_train_steps), desc='> Train epoch: ', leave=False)

        model.train()
        # 1. Train
        # @torch.compile()
        for batch_idx, batch_x in enumerate(train_dataloader):
            if batch_idx in subset_random:
                if cfg.laptop and subset_random_idx >= 1:
                    break
                # 1.1 Move batch to GPU
                batch_x = {k: v.to(cfg.device) for k,v in batch_x.items()} 

                # 1.2 Zero gradients for every batch
                optimizer.zero_grad()
                
                # 1.3 Forward pass
                outputs = model(**batch_x)

                # 1.4 Loss & Backpropagation
                train_loss = outputs['loss']
                train_loss.backward()

                # 1.5 Adjust learning weights
                optimizer.step()

                # 1.6 Logging (batch)
                sum_train_loss += train_loss
                log_str = (
                    f'\tepoch {epoch+1} ~ '
                    f'batch {subset_random_idx+1}/{epoch_train_steps}, '
                        f'train_loss: {train_loss:.4f}'
                )
                batch_progress_bar.update(1)
                batch_progress_bar.write(log_str)
                if cfg.logging and is_wandb_available():
                    wandb.log({'train_loss': train_loss})

                subset_random_idx += 1

        # 2. Update learning rate after each epoch
        lr_scheduler.step()
        t.cuda.empty_cache()

        # 3. Evaluation
        model.eval()
        val_logs = validate(model, cfg.device, vocab, epoch, valid_dataloader)
        if cfg.logging and is_wandb_available():
            wandb.log(val_logs)

        # 4. Save model(s)
        # 4.1 Save best model
        if (
            (eval_target is None)
            or (cfg.target_metric_bigger_better and eval_target <= val_logs[cfg.target_metric])
            or (not cfg.target_metric_bigger_better and eval_target >= val_logs[cfg.target_metric])
        ):
            # Pretty Printing
            if eval_target is None:
                eval_target = 0.0
            print(
                f'\nUpdating the model with better {cfg.target_metric}.\n'
                f'Prev: {eval_target:.4f}, Curr (epoch={epoch}): {val_logs[cfg.target_metric]:.4f}\n'
                f'Removing the previous best checkpoint.\n'
            )
            eval_target = val_logs[cfg.target_metric]
            t.save(model.state_dict(), best_ckpt_path / 'pytorch_model.bin')
            pickle.dump(model.get_config(), open(best_ckpt_path / 'model_cfg.pkl', 'wb'))


        # 4.2 Save model after each epoch
        if cfg.save_all_epochs:
            t.save(model.state_dict(), all_ckpt_path / f'e{epoch:04d}.bin')
            if not os.path.exists(all_ckpt_path / 'model_cfg.pkl'):
                pickle.dump(model.get_config(), open(all_ckpt_path / 'model_cfg.pkl', 'wb'))

        # 4.3 Save last model
        t.save(optimizer.state_dict(), last_ckpt_path / 'optimizer.pt')
        t.save({'last_epoch': cfg.num_epochs}, last_ckpt_path / 'scheduler.pt')
        t.save(model.state_dict(), last_ckpt_path / 'pytorch_model.bin')
        pickle.dump(model.get_config(), open(last_ckpt_path / 'model_cfg.pkl', 'wb'))

        # 4.4 Logging (epoch)
        mean_train_loss = sum_train_loss / epoch_train_steps
        log_str = (
            f'Epoch {epoch+1}/{cfg.num_epochs} -> '
            f'lr: {t.tensor(optimizer.param_groups[0]["lr"])}| '
            f'mean_train_loss: {mean_train_loss}| '
            f'mean_val_loss: {val_logs["mean_val_loss"]}| '
            f'val_per: {val_logs["mean_val_per"]}'
        )
        epoch_progress_bar.update(1)
        epoch_progress_bar.write(log_str)
        if cfg.logging and is_wandb_available():
            wandb.log({
                'lr': t.tensor(optimizer.param_groups[0]["lr"]),
                'mean_train_loss': mean_train_loss,
            })



def validate(
        model, device, vocab, epoch, 
        validate_dataloader, log_step=100
):
    val_losses = []
    val_edit_distances, val_num_phonemes = [], []
    phoneme_labels, phoneme_predictions = [], []

    validate_steps = int(len(list(enumerate(validate_dataloader))) / log_step)
    val_progress_bar = tqdm.tqdm(iterable=range(validate_steps), desc='> Validate epoch: ', leave=False)

    # For validation_dataloader, batch-size == 1
    for batch_idx, batch_x in enumerate(validate_dataloader):
        if cfg.laptop:
            if batch_idx >= 1:
                break
        with t.no_grad():
            # Ground Truth
            phoneme_label = batch_x['phoneme_labels'].numpy()[0]
            phoneme_label_ipa = idx_phonemes(vocab, phoneme_label)
            phoneme_labels.append(phoneme_label_ipa)
            # Predictions
            batch_x = {k: v.to(device) for k,v in batch_x.items()}
            outputs = model(**batch_x)
            phoneme_logits = outputs['phoneme_logits'].cpu().numpy()
            phoneme_prediction = _ctc_decode(vocab, phoneme_logits)
            phoneme_prediction_ipa = idx_phonemes(vocab, phoneme_prediction)
            # Loss
            val_loss = outputs['loss'].item()
            val_losses.append(val_loss)
            # PER
            val_edit_distance = editdistance.eval(phoneme_label, phoneme_prediction)
            val_edit_distances.append(val_edit_distance)
            val_num_phoneme = len(phoneme_label)
            val_num_phonemes.append(val_num_phoneme)
            val_per = val_edit_distance / val_num_phoneme

            # Logging
            if (batch_idx % log_step == 0) or (batch_idx == len(validate_dataloader)):
                log_str = (
                    f'\n\tepoch {epoch+1}, validation file: {batch_idx+1}/{len(validate_dataloader)}:'
                    f'\n\t  validation metrics ~ '
                    f'val_loss: {val_loss} | '
                    f'edit_distance: {val_edit_distance} | '
                    f'phonemes: {val_num_phoneme} | '
                    f'val_per: {round(val_per * 100, 2)} |\n'
                    f'\t  true_label: {phoneme_label_ipa}\n'
                    f'\t  pred_label: {phoneme_prediction_ipa}'
                )
                val_progress_bar.update(1)
                val_progress_bar.write(log_str)

    return {
        'mean_val_per': np.sum(val_edit_distances) / np.sum(val_num_phonemes),
        'mean_val_loss': np.array(val_losses).mean()
    }



def test(model, device, vocab, test_dl, dataset_name, log_step=100, laptop=False):
    """
    Very similar to validate().
    """
    test_edit_distances, test_num_phonemes = [], []
    phoneme_labels, phoneme_predictions = [], []

    test_steps = int(len(list(enumerate(test_dl))) / log_step)
    test_progress_bar = tqdm.tqdm(
        iterable=range(test_steps),
        desc=f'> Test ({dataset_name}) epoch: ',
        leave=False
    )

    # For testing: batch-size == 1
    for batch_idx, batch_x in enumerate(test_dl):
        if laptop:
            if batch_idx >= 1:
                break
        with t.no_grad():
            # Ground Truth
            phoneme_label = batch_x['phoneme_labels'].numpy()[0]
            phoneme_label_ipa = idx_phonemes(vocab, phoneme_label)
            phoneme_labels.append(phoneme_label_ipa)
            # Predictions
            batch_x = {k: v.to(device) for k,v in batch_x.items()}
            outputs = model(**batch_x)
            phoneme_logits = outputs['phoneme_logits'].cpu().numpy()
            phoneme_prediction = _ctc_decode(vocab, phoneme_logits)
            phoneme_prediction_ipa = idx_phonemes(vocab, phoneme_prediction)
            # PER
            test_edit_distance = editdistance.eval(phoneme_label, phoneme_prediction)
            test_edit_distances.append(test_edit_distance)
            test_num_phoneme = len(phoneme_label)
            test_num_phonemes.append(test_num_phoneme)
            test_per = test_edit_distance / test_num_phoneme

            # Logging
            if (batch_idx % log_step == 0) or (batch_idx == len(test_dl)):
                log_str = (
                    f'\n\t  {dataset_name} test metrics ~ '
                    f'edit_distance: {test_edit_distance} | '
                    f'num_phonemes: {test_num_phoneme} | '
                    f'per: {round(test_per * 100, 2)} |\n'
                    f'\t  true_label: {phoneme_label_ipa}\n'
                    f'\t  pred_label: {phoneme_prediction_ipa}'
                )
                test_progress_bar.update(1)
                test_progress_bar.write(log_str)

    return {
        'mean_test_per': np.sum(test_edit_distances) / np.sum(test_num_phonemes),
    }











# ################################
# Main
# ################################
if __name__ == '__main__':
    # 1. config & logging
    cfg = parse_args()
    if cfg.logging:
        init_logger(cfg, 'phoneme_recognizer')

    # 2. create meta files/dirs
    pickle.dump(cfg, open(cfg.exp_dir / 'experiment_args.pkl', 'wb'))
    best_ckpt_path = cfg.exp_dir / 'best-model-ckpt'
    last_ckpt_path = cfg.exp_dir / 'last-model-ckpt'
    all_ckpt_path = cfg.exp_dir / 'model-ckpts'

    os.mkdir(best_ckpt_path)
    os.mkdir(last_ckpt_path)
    if cfg.save_all_epochs:
        os.mkdir(all_ckpt_path)
        
    # 3. datasplits/dataloaders, model, optimizer, scheduler
    vocab, train_dl, valid_dl, test_dl = _prepare_datasets(
        cfg.exp_dir,
        cfg.batch_size,
        pd.read_csv(cfg.cp_csv_path),
        cfg.cropping,
        cfg.train_from_ckpt,
    )
    # model, optimizer, scheduler
    model, optimizer, lr_scheduler = _prepare_model_optimizer_scheduler(
        cfg,
        vocab
    )

    # 4. print some relevant stuff for training
    print('\nTraining started ...\n')
    print('training config:\n', cfg, '\n')
    print(f'device used: {cfg.device}')
    print('\nvocab', vocab)
    print('\ndataset files used for training: ', len(train_dl) * cfg.batch_size)
    print(f'batch_size = {cfg.batch_size}')
    print('len(train_dataloader) -> batched dataset', len(train_dl))
    print(f'Randomly use {cfg.samples_per_epoch} samples per epoch during training.')
    print(f'This will result in {cfg.samples_per_epoch} / {cfg.batch_size} = {int(cfg.samples_per_epoch / cfg.batch_size)} batches per epoch')
    print('\ntrainable parameters: ', count_parameters(model), '\n')
    print('... starting training ... ')

    # 5. train & validation loop
    train(
        cfg, model, optimizer, lr_scheduler,
        vocab, train_dl, valid_dl,
        best_ckpt_path=best_ckpt_path,
        last_ckpt_path=last_ckpt_path,
        all_ckpt_path=all_ckpt_path
    )

    # 6. test best model
    # load best model
    model_cfg = pickle.load(open(best_ckpt_path / 'model_cfg.pkl', 'rb'))
    best_model = Wav2Vec2_PR(
        model_cfg['pretrain_cfg'],
        model_cfg['cache_dir'],
        model_cfg['huggingface_model_id']
    ).to(cfg.device)
    best_ckpt = t.load(best_ckpt_path / 'pytorch_model.bin')
    best_model.load_state_dict(best_ckpt)
    
    # CommonPhone test split
    cp_test_results = test(best_model, cfg.device, vocab, test_dl, 'cp_test', laptop=cfg.laptop)
    cp_test_results['mean_cp_test_per'] = cp_test_results['mean_test_per']
    del cp_test_results['mean_test_per']

    # HPRC 
    hprc_df = pd.read_csv(cfg.hprc_csv_path)
    hprc_df.to_csv(cfg.exp_dir / 'hprcs.csv', index=False)

    # HPRC - Normal rate test set
    hprcN_dl = _get_dataset('hprcN', vocab, hprc_df, 1, cropping=False)
    hprcN_results = test(best_model, cfg.device, vocab, hprcN_dl, 'hprcN', laptop=cfg.laptop)
    hprcN_results['mean_hprcN_per'] = hprcN_results['mean_test_per']
    del hprcN_results['mean_test_per']

    # HPRC - Fast rate test set
    hprcF_dl = _get_dataset('hprcF', vocab, hprc_df, 1, cropping=False)
    hprcF_results = test(best_model, cfg.device, vocab, hprcF_dl, 'hprcF', laptop=cfg.laptop)
    hprcF_results['mean_hprcF_per'] = hprcF_results['mean_test_per']
    del hprcF_results['mean_test_per']

    print('\n\n ~ TEST RESULTS ~\n')
    print(f'CommonPhone (Test) mean PER: {round(cp_test_results["mean_cp_test_per"] * 100, 2)}% for {len(test_dl)} files.\n')
    print(f'HPRC (Normal) mean PER: {round(hprcN_results["mean_hprcN_per"] * 100, 2)}% for {len(hprcN_dl)} files.\n')
    print(f'HPRC (Fast) mean PER: {round(hprcF_results["mean_hprcF_per"] * 100, 2)}% for {len(hprcF_dl)} files.\n')
    if cfg.logging and is_wandb_available():
        wandb.log(cp_test_results)
        wandb.log(hprcN_results)
        wandb.log(hprcF_results)



