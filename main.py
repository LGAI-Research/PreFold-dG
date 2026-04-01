import argparse
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import load_dataset, load_group_map, load_split, random_split
from src.model import PreFolddG
from src.utils import auroc, mae, pcc, precision, recall, rmse, srcc


def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0.0
    device = next(model.parameters()).device

    for _, mut, wt in data_loader:
        mut_x, mut_y = mut
        wt_x, wt_y = wt
        mut_y = mut_y.to(device)
        wt_y = wt_y.to(device)
        ddg_y = mut_y - wt_y

        mut_h = model(mut_x.to(device))
        wt_h = model(wt_x.to(device))
        ddg_h = mut_h - wt_h

        loss = (F.mse_loss(mut_h.squeeze(1), mut_y)
                + F.mse_loss(wt_h.squeeze(1), wt_y)
                + F.mse_loss(ddg_h.squeeze(1), ddg_y))

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss


def predict(model, data_loader):
    model.eval()
    index = []
    pred_dg = []
    pred_ddg = []
    true_dg = []
    true_ddg = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, mut, wt in data_loader:
            mut_x, mut_y = mut
            wt_x, wt_y = wt
            mut_y = mut_y.to(device)
            wt_y = wt_y.to(device)
            ddg_y = mut_y - wt_y

            mut_h = model(mut_x.to(device))
            wt_h = model(wt_x.to(device))
            ddg_h = mut_h - wt_h

            index.extend(i)
            pred_dg.extend(mut_h.detach().cpu().flatten())
            pred_ddg.extend(ddg_h.detach().cpu().flatten())
            true_dg.extend(mut_y.detach().cpu().flatten())
            true_ddg.extend(ddg_y.detach().cpu().flatten())

    dataset = data_loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    group = dataset.get_pair_groups(index)

    reorder = np.argsort(index)
    return {
        'pred_dg': np.array(pred_dg)[reorder],
        'pred_ddg': np.array(pred_ddg)[reorder],
        'true_dg': np.array(true_dg)[reorder],
        'true_ddg': np.array(true_ddg)[reorder],
        'group': group[reorder],
    }


def evaluate(preds, *, group_by=lambda t: t, group_min_size=2):
    h, y = preds['pred_ddg'], preds['true_ddg']

    group = np.vectorize(group_by)(preds['group'])
    group_pcc = []
    group_srcc = []

    for g in np.unique(group):
        idx = np.where(group == g)[0]
        if len(idx) < group_min_size:
            continue
        group_pcc.append(pcc(h[idx], y[idx]))
        group_srcc.append(srcc(h[idx], y[idx]))

    return {
        'group_pcc': np.mean(group_pcc),
        'group_srcc': np.mean(group_srcc),
        'pcc': pcc(h, y),
        'srcc': srcc(h, y),
        'rmse': rmse(h, y),
        'mae': mae(h, y),
        'auroc': auroc(h, y),
    }


def make_model(args, device):
    model = PreFolddG(
        si_encoder_hidden_dim=args.s_inputs_encoder_hidden_dim,
        si_encoder_dropout=args.s_inputs_encoder_dropout,
        s_encoder_hidden_dim=args.s_encoder_hidden_dim,
        s_encoder_dropout=args.s_encoder_dropout,
        z_encoder_dropout=args.z_encoder_dropout,
        embed_dim=args.embed_dim,
        hidden_dim=args.predictor_hidden_dim).to(device)
    return model


def format_scores(scores):
    return '  '.join(f'{k}={v:>7.4f}' for k, v in scores.items())


def main(args):
    print(args)
    device = torch.device(args.device)

    train_dataset = load_dataset(args.train_data)

    train_folds = getattr(args, 'train_folds', None)
    test_fold = getattr(args, 'test_fold', None)
    test_data = getattr(args, 'test_data', None)
    group_map_path = getattr(args, 'group_map', None)
    group_min_size = getattr(args, 'group_min_size', 2)

    # group mapping
    if group_map_path:
        gmap = load_group_map(group_map_path)
        group_by = lambda pdb: gmap.get(pdb, pdb)
    else:
        group_by = lambda t: t

    # test dataset
    if test_data:
        test_dataset = load_dataset(test_data)

    # determine runs
    pdb_list = train_dataset.get_pair_groups(range(len(train_dataset)))
    split_path = getattr(args, 'split', None)

    if split_path:
        folds = load_split(split_path, pdb_list)
    else:
        folds = random_split(pdb_list, n_folds=3, seed=getattr(args, 'seed', None))

    if train_folds is None:
        train_folds = list(np.unique(folds[folds >= 0]))

    runs = train_folds

    all_scores = []

    for fold in runs:
        # train loader
        if fold is not None:
            remaining = [f for f in train_folds if f != fold]
            train_idx = np.where(np.isin(folds, remaining))[0]
            train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # test loader
        if test_data:
            test_loader = DataLoader(test_dataset, batch_size=1)
        elif test_fold is not None:
            test_idx = np.where(folds == test_fold)[0]
            test_loader = DataLoader(Subset(train_dataset, test_idx), batch_size=1)
        elif fold is not None:
            test_idx = np.where(folds == fold)[0]
            test_loader = DataLoader(Subset(train_dataset, test_idx), batch_size=1)
        else:
            test_loader = None

        if len(runs) > 1:
            print(f'\n=== fold {fold} ===')

        model = make_model(args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            loss = train(model, train_loader, optimizer)

            if test_loader:
                preds = predict(model, test_loader)
                scores = evaluate(preds, group_by=group_by, group_min_size=group_min_size)
                print(f'[epoch {epoch:3d}]  loss={loss:>8.4f}  {format_scores(scores)}')
            else:
                print(f'[epoch {epoch:3d}]  loss={loss:>8.4f}')

        if test_loader:
            all_scores.append(scores)

    if len(all_scores) > 1:
        mean_scores = {k: np.mean([s[k] for s in all_scores]) for k in all_scores[0]}
        print(f'\n=== mean ===')
        print(format_scores(mean_scores))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    # model
    parser.add_argument('--s_inputs_encoder_hidden_dim', type=int)
    parser.add_argument('--s_inputs_encoder_dropout', type=float)
    parser.add_argument('--s_encoder_hidden_dim', type=int)
    parser.add_argument('--s_encoder_dropout', type=float)
    parser.add_argument('--z_encoder_dropout', type=float)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--predictor_hidden_dim', type=int)

    # training
    parser.add_argument('--device', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    # data
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--train_folds', type=int, nargs='+')
    parser.add_argument('--test_fold', type=int)
    parser.add_argument('--group_map', type=str)
    parser.add_argument('--group_min_size', type=int)

    args = parser.parse_args()

    # Load config file as base defaults
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # CLI overrides: only update config with explicitly provided CLI args
    cli_args = {k: v for k, v in vars(args).items() if k != 'config' and v is not None}
    config.update(cli_args)

    return SimpleNamespace(**config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
