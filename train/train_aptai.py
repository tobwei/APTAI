import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.integrations import is_wandb_available
import wandb
import tqdm
import argparse
import os
from pathlib import Path
from datetime import datetime
import json
import pickle
import pandas as pd
import editdistance
import random
import aptai
from dataset_hprc import(
    HPRCDataset,
)
from utility import(
    count_parameters,
    tvs_metric_ppc,
    tvs_metric_rmse,
    evaluate_overlap,
    get_stats,
    phn_frame_id2phn,
    dict_to_csv,
    init_logger,
)




# ################################
# Arguments
# ################################
def parse_args():
    parser = argparse.ArgumentParser(
        description = ('Combining AAI and PTA (Acoustic Phoneme To Articulatory Inversion.')
    )

    # Dirs, Paths, Misc
    parser.add_argument(
        '--exp_dir', type=Path, default=None,
        help='If provided continue training, otherwise start new training run.'

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
        '--hprc_prep_csv_path', type=Path, default=Path('../data/HPRC_prep/hprc.csv'),
        help='Contains the prepared hrpc dataset.'
    )
    parser.add_argument(
        '--vocab_path', type=Path, default='../data/vocab.json',
        help='Path to the vocab used for the w2v2 phoneme recognizer.'
    )

    # Training
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
        '--target_metric', type=str, default="val_mean_loss",
        help='Possible: val_mean_loss, val_mean_rmse, val_mean_pcc'
    )
    parser.add_argument(
        '--target_metric_bigger_better', action=argparse.BooleanOptionalAction,
        help='For certain metrics, bigger values equal better performance (e.g. val_accuracy)'
    )
    parser.add_argument(
        '--train_val_rate', type=str, default='both',
        help='Possible: "N", "F", "both".'
    )

    # Model specific
    parser.add_argument(
        '--huggingface_model_id', type=str, default='facebook/wav2vec2-large-xlsr-53',
        help='The huggingface pre-trained model that is used.'
    )
    parser.add_argument(
        '--tv_drop', type=float, default=0.1,
        help= ('Amount of dropout applied in the tv head of aptai.')
    )
    parser.add_argument(
        '--phn_drop', type=float, default=0.1,
        help= ('Amount of dropout applied in the phn head of aptai.')
    )

    # Finalize arugments
    args = parser.parse_args()
    args.date_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.exp_name = (
        f'{args.prefix}'
        f'_e{args.num_epochs}'
        f'_bs{args.batch_size}'
        f'_rate:{args.train_val_rate}'
    )
    if args.laptop:
        args.num_epochs = 1
        args.num_warmup_epochs = 1
        args.num_static_epochs = 1
        
    assert os.path.exists(args.vocab_path)
    file = open(args.vocab_path)
    args.vocab = json.load(file)
    file.close()

    if args.exp_dir is None:
        # Train from scratch
        args.exp_dir = Path(f'../experiments/APTAI/{args.date_time}_{args.exp_name}')
        args.exp_dir.mkdir(parents=True, exist_ok=False)
        args.train_from_ckpt = False
    else:
        # Train from checkpoint
        args.exp_dir = Path(args.exp_dir)
        assert args.exp_dir.exists()
        args.train_from_ckpt = True

    return args



# ################################
# Training preparation
# ################################
def _prepare_datasets(
        hprc_df,
        hprc_spks,
        test_spk,
        cfg,
):
    """
    Important note: hprc.csv has no repetitions per utterance and speaker. 
    
    df: is the read .csv file which represents the dataset (HPRC)

    spks - test_spk: 90% for train, 10% for valid at random, constrained
    by the fact that only unseen (druing training) utterances are used
    for valid split.
    highly dependent on: train_rate, valid_rate, and test_rate args

    ---
    test always has two dataloaders: N and F rates.
    train_val can have: N, F, both rates with 90 and 10% train/val splits.
    """
    test_spk_df = hprc_df[hprc_df.speaker == test_spk] 
    
    # final test_df's 
    test_f_df = test_spk_df[test_spk_df.rate == 'F']
    test_n_df = test_spk_df[test_spk_df.rate == 'N']

    # hprc data from remaining speakers
    hprc_df_notest = hprc_df.drop(test_spk_df.index)
    hprc_spks_notest = [spk for spk in hprc_spks if spk != test_spk]
    train_list = []
    valid_list = []
    
    # select 10% of the text to be used for validation
    hprc_text = hprc_df_notest.text.unique()
    valid_text = random.choices(hprc_text, k=int(len(hprc_text)*0.1))

    for spk in hprc_spks_notest:
        hprc_df_spk = hprc_df_notest[hprc_df_notest.speaker == spk]
        tmp_valid = hprc_df_spk[hprc_df_spk.text.isin(valid_text)]
        tmp_train = hprc_df_spk.drop(tmp_valid.index)
        train_list.append(tmp_train)
        valid_list.append(tmp_valid)
    train_df = pd.concat(train_list, ignore_index=True)
    valid_df = pd.concat(valid_list, ignore_index=True)

    # final train_df
    if cfg.train_val_rate == 'both':
        train_df = train_df
    elif cfg.train_val_rate == 'N' or cfg.train_val_rate == 'F':
        train_df = train_df[train_df.rate == cfg.train_val_rate]
    else:
        raise ValueError
    
    # final valid_df
    if cfg.train_val_rate == 'both':
        valid_df = valid_df
    elif cfg.train_val_rate == 'N' or cfg.train_val_rate == 'F':
        valid_df = valid_df[valid_df.rate == cfg.train_val_rate]
    else:
        raise ValueError

    # dataloaders based on previous data splits
    train_dl = _get_dataset('hprc', cfg.vocab, train_df, cfg.batch_size, shuffle=True)
    valid_dl = _get_dataset('hprc', cfg.vocab, valid_df, batch_size=1, shuffle=False)
    test_f_dl = _get_dataset('hprc', cfg.vocab, test_f_df, batch_size=1, shuffle=False)
    test_n_dl = _get_dataset('hprc', cfg.vocab, test_n_df, batch_size=1, shuffle=False)

    return train_dl, valid_dl, test_n_dl, test_f_dl


def _get_dataset(dataset, vocab, df, batch_size, **kwargs):
    assert dataset in ['hprc', 'xrmb']
    if dataset == 'hprc':
        return torch.utils.data.DataLoader(
            HPRCDataset(df, vocab, rate='both'),
            batch_size=batch_size,
            collate_fn=_collate_fn,
            pin_memory=True,
            **kwargs,
        )
            
    elif dataset == 'xrmb':
        return torch.utils.data.DataLoader(

        )


def _collate_fn(batch):
    """
    49hz norm TV variant.
    """
    # print(batch)
    return {
        'audio_inputs': torch.nn.utils.rnn.pad_sequence(
            [x['audio'] for x in batch],
            batch_first=True,
            padding_value=0.0,
        ),
        'audio_lengths': torch.LongTensor(
            [x['audio_len'] for x in batch],
        ),
        'phn_frames_49hz' : torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(x['phn_frames_49hz']) for x in batch],
            batch_first=True,
            padding_value=0,
        ),
        'LA': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['LA']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'LP': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['LP']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'JA': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['JA']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'TTCL': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['TTCL']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'TTCD': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['TTCD']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'TMCL': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['TMCL']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'TMCD': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['TMCD']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'TBCL': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['TBCL']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
        'TBCD': torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x['tvs_norm_49hz']['TBCD']) for x in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
    }


def _prepare_model_optim_scheduler(args_cfg):
    pretrain_cfg, *_ = transformers.PretrainedConfig.get_config_dict(
        args_cfg.huggingface_model_id
    )
    pretrain_cfg['vocab_size'] = len(args_cfg.vocab)
    pretrain_cfg = transformers.Wav2Vec2Config.from_dict(pretrain_cfg)
    
    model = aptai.APTAI(
        device=args_cfg.device,
        vocab=args_cfg.vocab,
        huggingface_model_id = args_cfg.huggingface_model_id,
        pretrain_cfg = pretrain_cfg,
        cache_dir = args_cfg.cache_dir,
    ).to(args_cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args_cfg.learning_rate,
        betas=(args_cfg.adam_beta1, args_cfg.adam_beta2),
        eps=args_cfg.adam_epsilon,
        weight_decay=args_cfg.adam_weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=_get_lr_schedule(
            args_cfg.num_warmup_epochs,
            args_cfg.num_static_epochs,
            args_cfg.lr_decay,
        ),
    )

    return model, optimizer, lr_scheduler


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


# ################################
# Train, Val, Test 
# ################################
def train(
    cfg, model, optimizer, lr_scheduler,
    train_dataloader, valid_dataloader,
    test_spk, best_ckpt_path,
    ):
    start_epoch = 0
    eval_target = None
    epoch_progress_bar = tqdm.tqdm(
        iterable=range(cfg.num_epochs),
        desc='Epochs: ',
        leave=False,
    )
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_train_steps = len(train_dataloader)
        batch_progress_bar = tqdm.tqdm(
            iterable=range(epoch_train_steps),
            desc='> Train epoch: ',
            leave=False,
        )
        sum_train_loss = 0

        model.train()
        # 1. Train
        for batch_idx, batch_x in enumerate(train_dataloader):
            if cfg.laptop:
                if batch_idx >= 1:
                    break
            # debugging
            # if batch_idx >= 300:
            #     print('press a buton so you dont forget to remove this later')
            #     input()
            #     break
            
            # 1.1 Move batch to GPU
            batch_x = {k: v.to(cfg.device) for k,v in batch_x.items()}

            # 1.2 Zero gradients for every batch
            optimizer.zero_grad()
            
            # 1.3 Forward pass
            outputs = model(epoch, **batch_x)

            # 1.4 Loss & Backpropagation
            train_loss = outputs['loss']
            train_mse_loss = outputs['mse_loss']
            train_ce_loss = outputs['ce_loss']
            train_loss.backward()

            # 1.5 Adjust learning weights
            optimizer.step()

            # 1.6 Logging (batch)
            sum_train_loss += train_loss
            curr_lr = optimizer.param_groups[0]['lr']
            log_str = (
                f'\tepoch {epoch+1} ~ '
                f'batch {batch_idx+1}/{epoch_train_steps}, '
                f'train_loss: {train_loss:.4f}, '
                f'train_mse_loss: {train_mse_loss:.4f}, '
                f'train_ce_loss: {train_ce_loss:.4f}, '
                f'lr: {curr_lr:.6f}'
            )
            batch_progress_bar.update(1)
            batch_progress_bar.write(log_str)
            if cfg.logging and is_wandb_available():
                wandb.log({'train_loss': train_loss})
                wandb.log({'train_mse_loss': train_mse_loss})
                wandb.log({'train_ce_loss': train_ce_loss})
                
        # 2. Update learning rate after each epoch
        lr_scheduler.step()
        torch.cuda.empty_cache()
        
        # 3. Evaluation after each epoch
        model.eval()
        val_logs = validate(
            model, cfg.device, cfg.vocab, epoch,
            cfg.exp_dir, test_spk, valid_dataloader
        )
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
            torch.save(model.state_dict(), best_ckpt_path / 'pytorch_model.bin')
            pickle.dump(model.get_config(), open(best_ckpt_path / 'model_cfg.pkl', 'wb'))
        # # 4.2 Save last model
        # t.save(optimizer.state_dict(), last_ckpt_path / 'optimizer.pt')
        # t.save({'last_epoch': cfg.num_epochs}, last_ckpt_path / 'scheduler.pt')
        # t.save(model.state_dict(), last_ckpt_path / 'pytorch_model.bin')
        # pickle.dump(model.get_config(), open(last_ckpt_path / 'model_cfg.pkl', 'wb'))

        # 4.4 Logging (epoch)
        mean_train_loss = sum_train_loss / epoch_train_steps
        # 4.4.1 console
        log_str = (
            f'Epoch {epoch+1}/{cfg.num_epochs} -> '
            f'lr: {torch.tensor(optimizer.param_groups[0]["lr"])}| '
            f'mean_train_loss: {mean_train_loss:.4f} | '
            f'val_mean_loss: {val_logs["val_mean_loss"]:.4f} | '
            f'val_mean_rmse: {val_logs["val_mean_rmse"]:.4f} | '
            f'val_mean_pcc: {val_logs["val_mean_pcc"]:.4f} |'
            f'val_mean_FER: {val_logs["val_mean_FER"]:.4f} | '
            f'val_mean_PER: {val_logs["val_mean_PER"]:.4f} | '
            f'val_mean_F1: {val_logs["val_mean_F1"]:.4f} | '
            f'val_mean_Rval: {val_logs["val_mean_Rval"]:.4f} | '
            f'val_mean_overlap: {val_logs["val_mean_overlap"]:.4f} | '
        )
        epoch_progress_bar.update(1)
        epoch_progress_bar.write(log_str)
        # 4.4.2 wandb
        if cfg.logging and is_wandb_available():
            wandb.log({
                'lr': torch.tensor(optimizer.param_groups[0]["lr"]),
                'train_mean_loss': mean_train_loss,
                'val_mean_loss': val_logs['val_mean_loss'],
                'val_mean_rmse': val_logs['val_mean_rmse'],
                'val_mean_pcc': val_logs['val_mean_pcc'],
                'val_mean_FER': val_logs['val_mean_FER'],
                'val_mean_overlap': val_logs['val_mean_overlap'],
                'val_mean_PER': val_logs['val_mean_PER']
            })



def validate(
    model, device, vocab, epoch, exp_dir, test_spk,
    val_dl, log_step=100,
    ):
    val_losses, val_rmses, val_pccs = [], [], []
    val_overlaps, val_ps, val_rs, val_f1s, val_rvals = [], [], [], [], []
    val_fc_edit_distances, val_fc_num_phns = [], []
    val_total_frames = 0
    val_corr_frames = 0
    
    val_steps = int(len(list(enumerate(val_dl))) / log_step)
    val_progress_bar = tqdm.tqdm(
        iterable=range(val_steps),
        desc='> Validate epoch: ',
        leave=False,
    )

    # For validation_dataloader, batch-size == 1
    for batch_idx, batch_x in enumerate(val_dl):
        if cfg.laptop:
            if batch_idx >= 5:
                break
        with torch.no_grad():
            # Ground Truth
            tvs_gt = torch.stack([batch_x['LA'], batch_x['LP'], batch_x['JA'],
                            batch_x['TTCL'], batch_x['TTCD'],
                            batch_x['TMCL'], batch_x['TTCD'],
                            batch_x['TBCL'], batch_x['TBCD']], dim=-1).float()
            
            # Predictions
            batch_x = {k: v.to(device) for k,v in batch_x.items()}
            outputs = model(epoch, **batch_x)
            tvs_pred = outputs['tvs_pred']
            
            # Loss
            val_loss = outputs['loss'].item()
            val_losses.append(val_loss)

            # Metrics
            # TVs
            tvs_gt = torch.squeeze(tvs_gt, dim=0).cpu().numpy()
            tvs_pred = torch.squeeze(tvs_pred, dim=0).cpu().numpy()

            # RMSE
            rmse = tvs_metric_rmse(tvs_gt, tvs_pred)
            mean_rmse = np.mean(list(rmse.values()))
            val_rmses.append(mean_rmse)
            
            # PCC
            pcc = tvs_metric_ppc(tvs_gt, tvs_pred)
            pcc_stats = [v.statistic for k,v in pcc.items()]
            pcc_pval = [v.pvalue for k,v in pcc.items()]
            mean_pcc = np.mean(pcc_stats)
            val_pccs.append(mean_pcc)

            # Phonemes
            gt_frames = batch_x['phn_frames_49hz']
            pred_frames = outputs['phn_fc_pred']
        
            # FER: 1 - (correct_pred_frames / total_frames)
            num_gt_frames = batch_x['phn_frames_49hz'].size(1)
            val_total_frames += num_gt_frames
            elem_equal = torch.eq(gt_frames, pred_frames)
            corr_frames = torch.sum(elem_equal).item()
            val_corr_frames += corr_frames
            cur_val_fer = 1 - (corr_frames / num_gt_frames)

            # overlap ( == 1 - FER)
            gt_f = gt_frames.detach().cpu().numpy()
            p_f = pred_frames.detach().cpu().numpy()
            overlap = evaluate_overlap(gt_f, p_f)
            val_overlaps.append(overlap)

            # boudary based: prec, rec, F1, R-Value with 20 ms tolerance!
            y = gt_frames.detach().cpu().numpy().squeeze()
            yhat = pred_frames.detach().cpu().numpy().squeeze()
            p, r, f1, rval = get_stats(y, yhat, tolerance=0.02) # 0.02
            val_ps.append(p)
            val_rs.append(r)
            val_f1s.append(f1)
            val_rvals.append(rval)

            # FC PER: frame groupby based
            yhat_grp = phn_frame_id2phn(yhat)
            y_grp = phn_frame_id2phn(y)
            val_fc_edit_dist = editdistance.eval(y_grp, yhat_grp)
            val_fc_num_phn = len(y_grp)
            val_fc_per = val_fc_edit_dist / val_fc_num_phn
            val_fc_edit_distances.append(val_fc_edit_dist)
            val_fc_num_phns.append(val_fc_num_phn)

            # Logging (single validation file related)
            if (batch_idx % log_step == 0) or (batch_idx == len(val_dl)):
                log_str = (
                    f'\n\ttest-speaker: {test_spk}, epoch {epoch+1}, '
                    f'validation file: {batch_idx+1}/{len(val_dl)}:'
                    f'\n\t ~ validation metrics ~ '
                    f'val_loss: {"{:.5}".format(val_loss)} | '
                    f'mean_RMSE: {"{:.3}".format(mean_rmse)} | '
                    f'mean_PCC: {"{:.3}".format(mean_pcc)} | '
                    f'FER: {"{:.3}".format(cur_val_fer)} |'
                    f'F1: {"{:.3}".format(f1)} | '
                    f'R-Val: {"{:.3}".format(rval)} | '
                    f'val_fc_PER: {round(val_fc_per * 100, 2)} |'
                )
                val_progress_bar.update(1)
                val_progress_bar.write(log_str)

    return {
        'val_mean_loss': np.array(val_losses).mean(),
        'val_mean_rmse': np.array(val_rmses).mean(),
        'val_mean_pcc': np.array(val_pccs).mean(),
        'val_mean_FER': 1 - (val_corr_frames / val_total_frames),
        'val_mean_PER': np.sum(val_fc_edit_distances) / np.sum(val_fc_num_phns),
        'val_mean_F1': np.array(val_f1s).mean(),
        'val_mean_p': np.array(val_ps).mean(),
        'val_mean_r': np.array(val_rs).mean(),
        'val_mean_Rval': np.array(val_rvals).mean(),
        'val_mean_overlap': np.mean(val_overlaps),
    }


def test(
    model, device, vocab, exp_dir, test_spk,
    test_dl, rate, log_step=100,
):
    assert rate in ['F', 'N']
    
    test_overlaps, test_ps, test_rs, test_f1s, test_rvals = [], [], [], [], []
    test_fc_edit_distances, test_fc_num_phns = [], []
    test_total_frames = 0
    test_corr_frames = 0
    test_rmse_TVs = {
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
    test_pcc_TVs = {
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

    test_steps = int(len(list(enumerate(test_dl))) / log_step)
    test_progress_bar = tqdm.tqdm(
        iterable=range(test_steps),
        desc=f'> Test ({test_spk} - {rate}) epoch: ',
        leave=False
    )

    # For test_dataloader, batch-size == 1
    for batch_idx, batch_x in enumerate(test_dl):
        if cfg.laptop:
            if batch_idx >= 1:
                break
        with torch.no_grad():
            # Ground Truth
            tvs_gt = torch.stack([batch_x['LA'], batch_x['LP'], batch_x['JA'],
                            batch_x['TTCL'], batch_x['TTCD'],
                            batch_x['TMCL'], batch_x['TTCD'],
                            batch_x['TBCL'], batch_x['TBCD']], dim=-1).float()
            
            # Predictions
            batch_x = {k: v.to(device) for k,v in batch_x.items()}
            outputs = model(cfg.num_epochs, **batch_x)
            tvs_pred = outputs['tvs_pred']

            # Metrics
            tvs_gt = torch.squeeze(tvs_gt, dim=0).cpu().numpy()
            tvs_pred = torch.squeeze(tvs_pred, dim=0).cpu().numpy()

            # Phonemes
            gt_frames = batch_x['phn_frames_49hz']
            pred_frames = outputs['phn_fc_pred']
            
            # FER: 1 - (correct_pred_frames / total_frames)
            num_gt_frames = batch_x['phn_frames_49hz'].size(1)
            test_total_frames += num_gt_frames
            elem_equal = torch.eq(gt_frames, pred_frames)
            corr_frames = torch.sum(elem_equal).item()
            test_corr_frames += corr_frames
            cur_test_fer = 1 - (corr_frames / num_gt_frames)

            # overlap ( == 1 - FER)
            gt_f = gt_frames.detach().cpu().numpy()
            p_f = pred_frames.detach().cpu().numpy()
            overlap = evaluate_overlap(gt_f, p_f)
            test_overlaps.append(overlap)

            # boudary based: prec, rec, F1, R-Value with 20 ms tolerance!
            y = gt_frames.detach().cpu().numpy().squeeze()
            yhat = pred_frames.detach().cpu().numpy().squeeze()
            p, r, f1, rval = get_stats(y, yhat, tolerance=0.02) # 0.02
            test_ps.append(p)
            test_rs.append(r)
            test_f1s.append(f1)
            test_rvals.append(rval)

            # FC PER: frame groupby based
            yhat_grp = phn_frame_id2phn(yhat)
            y_grp = phn_frame_id2phn(y)
            test_fc_edit_dist = editdistance.eval(y_grp, yhat_grp)
            test_fc_num_phn = len(y_grp)
            test_fc_per = test_fc_edit_dist / test_fc_num_phn
            test_fc_edit_distances.append(test_fc_edit_dist)
            test_fc_num_phns.append(test_fc_num_phn)

            # TVs
            # RSME
            rmse_per_tv = tvs_metric_rmse(tvs_gt, tvs_pred)
            mean_rmse = np.mean(list(rmse_per_tv.values()))
            test_rmse_TVs['LA'].append(rmse_per_tv['LA'])
            test_rmse_TVs['LP'].append(rmse_per_tv['LP'])
            test_rmse_TVs['JA'].append(rmse_per_tv['JA'])
            test_rmse_TVs['TTCL'].append(rmse_per_tv['TTCL'])
            test_rmse_TVs['TTCD'].append(rmse_per_tv['TTCD'])
            test_rmse_TVs['TMCL'].append(rmse_per_tv['TMCL'])
            test_rmse_TVs['TMCD'].append(rmse_per_tv['TMCD'])
            test_rmse_TVs['TBCL'].append(rmse_per_tv['TBCL'])
            test_rmse_TVs['TBCD'].append(rmse_per_tv['TBCD'])

            # PCC
            pcc_per_tv = tvs_metric_ppc(tvs_gt, tvs_pred)
            pcc_stats = [v.statistic for k,v in pcc_per_tv.items()]
            mean_pcc = np.mean(pcc_stats)
            test_pcc_TVs['LA'].append(pcc_stats[0])
            test_pcc_TVs['LP'].append(pcc_stats[1])
            test_pcc_TVs['JA'].append(pcc_stats[2])
            test_pcc_TVs['TTCL'].append(pcc_stats[3])
            test_pcc_TVs['TTCD'].append(pcc_stats[4])
            test_pcc_TVs['TMCL'].append(pcc_stats[5])
            test_pcc_TVs['TMCD'].append(pcc_stats[6])
            test_pcc_TVs['TBCL'].append(pcc_stats[7])
            test_pcc_TVs['TBCD'].append(pcc_stats[8])
            
    # for test(), also report mean values per TV
    mean_test_rmse_TVs = {
        'LA': np.array(test_rmse_TVs['LA']).mean(),
        'LP': np.array(test_rmse_TVs['LP']).mean(),
        'JA': np.array(test_rmse_TVs['JA']).mean(),
        'TTCL': np.array(test_rmse_TVs['TTCL']).mean(),
        'TTCD': np.array(test_rmse_TVs['TTCD']).mean(),
        'TMCL': np.array(test_rmse_TVs['TMCL']).mean(),
        'TMCD': np.array(test_rmse_TVs['TMCD']).mean(),
        'TBCL': np.array(test_rmse_TVs['TBCL']).mean(),
        'TBCD': np.array(test_rmse_TVs['TBCD']).mean(),
    }
    
    mean_test_pcc_TVs = {
        'LA': np.array(test_pcc_TVs['LA']).mean(),
        'LP': np.array(test_pcc_TVs['LP']).mean(),
        'JA': np.array(test_pcc_TVs['JA']).mean(),
        'TTCL': np.array(test_pcc_TVs['TTCL']).mean(),
        'TTCD': np.array(test_pcc_TVs['TTCD']).mean(),
        'TMCL': np.array(test_pcc_TVs['TMCL']).mean(),
        'TMCD': np.array(test_pcc_TVs['TMCD']).mean(),
        'TBCL': np.array(test_pcc_TVs['TBCL']).mean(),
        'TBCD': np.array(test_pcc_TVs['TBCD']).mean(),
    }

    return {
        f'test_{rate}_mean_rmse': np.array(list(mean_test_rmse_TVs.values())).mean(),
        f'test_{rate}_mean_pcc': np.array(list(mean_test_pcc_TVs.values())).mean(),

        f'test_{rate}_mean_LA_pcc': mean_test_pcc_TVs['LA'],
        f'test_{rate}_mean_LP_pcc': mean_test_pcc_TVs['LP'],
        f'test_{rate}_mean_JA_pcc': mean_test_pcc_TVs['JA'],
        f'test_{rate}_mean_TTCL_pcc': mean_test_pcc_TVs['TTCL'],
        f'test_{rate}_mean_TTCD_pcc': mean_test_pcc_TVs['TTCD'],
        f'test_{rate}_mean_TMCL_pcc': mean_test_pcc_TVs['TMCL'],
        f'test_{rate}_mean_TMCD_pcc': mean_test_pcc_TVs['TMCD'],
        f'test_{rate}_mean_TBCL_pcc': mean_test_pcc_TVs['TBCL'],
        f'test_{rate}_mean_TBCD_pcc': mean_test_pcc_TVs['TBCD'],
        # 'test_mean_pcc_TVs': mean_test_pcc_TVs,
        
        f'test_{rate}_mean_LA_rmse': mean_test_rmse_TVs['LA'],
        f'test_{rate}_mean_LP_rmse': mean_test_rmse_TVs['LP'],
        f'test_{rate}_mean_JA_rmse': mean_test_rmse_TVs['JA'],
        f'test_{rate}_mean_TTCL_rmse': mean_test_rmse_TVs['TTCL'],
        f'test_{rate}_mean_TTCD_rmse': mean_test_rmse_TVs['TTCD'],
        f'test_{rate}_mean_TMCL_rmse': mean_test_rmse_TVs['TMCL'],
        f'test_{rate}_mean_TMCD_rmse': mean_test_rmse_TVs['TMCD'],
        f'test_{rate}_mean_TBCL_rmse': mean_test_rmse_TVs['TBCL'],
        f'test_{rate}_mean_TBCD_rmse': mean_test_rmse_TVs['TBCD'],
        # 'test_mean_rmse_TVs': mean_test_rmse_TVs,

        f'test_{rate}_mean_FER': 1 - (test_corr_frames / test_total_frames),
        f'test_{rate}_mean_PER': np.sum(test_fc_edit_distances) / np.sum(test_fc_num_phns),
        f'test_{rate}_mean_overlap': np.mean(test_overlaps),
        f'test_{rate}_mean_F1': np.array(test_f1s).mean(),
        f'test_{rate}_mean_p': np.array(test_ps).mean(),
        f'test_{rate}_mean_r': np.array(test_rs).mean(),
        f'test_{rate}_mean_Rval': np.array(test_rvals).mean(),
    }









# ################################
# Main
# ################################
if __name__ == '__main__':
    """
    Leave-one-speaker-out loop for the 8 speakers of HPRC
        one speaker for testing
        remaning speakers: random 90% training, and 10% validation
    """
    # 1. config, logging, results_dir
    torch.cuda.empty_cache()
    cfg = parse_args()
    pickle.dump(cfg, open(cfg.exp_dir / 'experiment_args.pkl', 'wb'))
    metrics_best_test_path = cfg.exp_dir / 'test_metrics'
    os.mkdir(metrics_best_test_path)

    if cfg.logging:
        init_logger(cfg, 'APTAI')
        
    # 2. leave-one-speaker-out (LOSO) for testing loop (8 speakers total)
    loso_n_test_results = {} # key is test-speaker ID, e.g. M01
    loso_f_test_results = {} # key is test-speaker ID, e.g. M01
    hprc_df = pd.read_csv(cfg.hprc_prep_csv_path)
    hprc_spks = hprc_df['speaker'].unique().tolist()
    for loso_idx in range(0, 8):
        test_spk = hprc_spks[loso_idx]

        # 2.1 dirs
        test_spk_dir = cfg.exp_dir / f'{test_spk}'
        best_ckpt_path = test_spk_dir / f'best-model-ckpt-{test_spk}'
        # last_ckpt_path = test_spk_dir / f'last-model-ckpt-{test_spk}'
        os.mkdir(test_spk_dir)
        os.mkdir(best_ckpt_path)
        # os.mkdir(last_ckpt_path)
        
        # 2.2 datasplits/dataloaders, model, optimizer, scheduler
        train_dl, valid_dl, test_n_dl, test_f_dl = _prepare_datasets(
            hprc_df,
            hprc_spks,
            test_spk,
            cfg,
        )
        # model, optimizer, scheduler
        model, optimizer, lr_scheduler = _prepare_model_optim_scheduler(cfg)
        
        # 2.3 print some stuff
        print('\nTraining started ...\n')
        print('training config:\n', cfg, '\n')
        print(f'device used: {cfg.device}')
        print(f'batch_size = {cfg.batch_size}')
        print('\ntrainable parameters: ', count_parameters(model), '\n')
        
        print(f'Leave-one-speaker-out loop started -- {loso_idx} ... ')
        print(f'~~~ test speaker: {test_spk} ~~~')
        print((f'\tTraining files: {len(train_dl) * cfg.batch_size},'
              f' and len(train_dl): {len(train_dl)} (batch-size: {cfg.batch_size})'))
        print(f'\tValidation files: {len(valid_dl)}')
        print(f'\tTest (Normal rate) files: {len(test_n_dl)}')
        print(f'\tTest (Fast rate) files: {len(test_f_dl)}')

        # 2.4 train & validation loop
        train(
            cfg, model, optimizer, lr_scheduler,
            train_dl, valid_dl, test_spk,
            best_ckpt_path=best_ckpt_path,
        )

        # 2.5 test
        pretrain_cfg, *_ = transformers.PretrainedConfig.get_config_dict(
            cfg.huggingface_model_id
        )
        pretrain_cfg['vocab_size'] = len(cfg.vocab)
        pretrain_cfg = transformers.Wav2Vec2Config.from_dict(pretrain_cfg)
        
        # 2.5.1 load best model
        best_model = aptai.APTAI(
            device=cfg.device,
            vocab=cfg.vocab,
            huggingface_model_id = cfg.huggingface_model_id,
            pretrain_cfg = pretrain_cfg,
            cache_dir = cfg.cache_dir,
        ).to(cfg.device)
        
        best_ckpt = torch.load(best_ckpt_path / 'pytorch_model.bin')
        best_model.load_state_dict(best_ckpt)

        # 2.5.1 compute test metrics
        # normal rates
        test_n_logs = test(
            best_model,
            cfg.device,
            cfg.vocab,
            cfg.exp_dir,
            test_spk,
            test_n_dl,
            rate='N',
            log_step=100,
        )
        # fast rates
        test_f_logs = test(
            best_model,
            cfg.device,
            cfg.vocab,
            cfg.exp_dir,
            test_spk,
            test_f_dl,
            rate='F',
            log_step=100,
        )
        
        # 2.4.2 save test results
        loso_n_test_results[test_spk] = test_n_logs
        loso_f_test_results[test_spk] = test_f_logs
        csv_n_file_name = f'{test_spk}_test_n_metrics.csv'
        csv_f_file_name = f'{test_spk}_test_f_metrics.csv'
        csv_dir_path = metrics_best_test_path
        dict_to_csv(test_n_logs, csv_dir_path, csv_n_file_name)
        dict_to_csv(test_f_logs, csv_dir_path, csv_f_file_name)
        
        # 2.4.3 print test results
        print(f'\n\nTEST RESULTS (speaker: {test_spk})')
        print('\nNORMAL:')
        print(loso_n_test_results[test_spk])
        print('\nFAST:')
        print(loso_f_test_results[test_spk])

        # 2.4.4 log test results (only for normal rate)
        wandb_test_log = {
            'test_N_mean_rmse': test_n_logs['test_N_mean_rmse'],
            'test_N_mean_pcc': test_n_logs['test_N_mean_pcc'],
            'test_N_mean_PER': test_n_logs['test_N_mean_PER'],
            'test_N_mean_FER': test_n_logs['test_N_mean_FER'],
            'test_N_mean_overlap': test_n_logs['test_N_mean_overlap'],
            'test_N_mean_F1': test_n_logs['test_N_mean_F1'],
            'test_N_mean_Rval': test_n_logs['test_N_mean_Rval'],
            
            'test_F_mean_rmse': test_f_logs['test_F_mean_rmse'],
            'test_F_mean_pcc': test_f_logs['test_F_mean_pcc'],
            'test_F_mean_PER': test_f_logs['test_F_mean_PER'],
            'test_F_mean_FER': test_f_logs['test_F_mean_FER'],
            'test_F_mean_overlap': test_f_logs['test_F_mean_overlap'],
            'test_F_mean_F1': test_f_logs['test_F_mean_F1'],
            'test_F_mean_Rval': test_f_logs['test_F_mean_Rval'],
        }
        if cfg.logging and is_wandb_available():
            wandb.log(wandb_test_log)
            
        # 2.5 elease memory
        torch.cuda.empty_cache()

    # 3. LOSO results (mean's across all test speakers)
    loso_n_csv_file_name = f'LOSO_N_mean_std_test_metrics.csv'
    loso_f_csv_file_name = f'LOSO_F_mean_std_test_metrics.csv'
    loso_f_mean_std_results = {} # key is metric
    loso_n_mean_std_results = {} # key is metric
    
    # same metrics for all test speakers
    # normal
    metrics = loso_n_test_results[random.choice(list(loso_n_test_results.keys()))].keys()
    for metric in metrics:
        tmp_metric_values = []
        for spk, results in loso_n_test_results.items():
            tmp_metric_values.append(results[metric])
        mean = np.mean(tmp_metric_values)
        std = np.std(tmp_metric_values)
        loso_n_mean_std_results[metric] = (mean, std)
    # fast
    metrics = loso_f_test_results[random.choice(list(loso_f_test_results.keys()))].keys()
    for metric in metrics:
        tmp_metric_values = []
        for spk, results in loso_f_test_results.items():
            tmp_metric_values.append(results[metric])
        mean = np.mean(tmp_metric_values)
        std = np.std(tmp_metric_values)
        loso_f_mean_std_results[metric] = (mean, std)
        # loso_f_mean_results[metric] = np.mean(tmp_metric_values)
    # save them as a file
    dict_to_csv(loso_f_mean_std_results, metrics_best_test_path, loso_f_csv_file_name)
    dict_to_csv(loso_n_mean_std_results, metrics_best_test_path, loso_n_csv_file_name)

    # 3.1 print LOSO mean results
    print('\n\nLOSO MEAN TEST RESULTS:')
    print('\nNORMAL:')
    print(loso_n_mean_std_results)
    print('\nFAST:')
    print(loso_f_mean_std_results)
    print('\n\n-> training done.\n\n')

    
