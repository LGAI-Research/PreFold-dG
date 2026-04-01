import numpy as np
import torch
import csv
import joblib


def load_split(csv_path, pdb_list):
    pdb_to_fold = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_to_fold[row['PDB']] = int(row['fold'])

    folds = [pdb_to_fold.get(pdb, -1) for pdb in pdb_list]
    return np.array(folds)


def load_group_map(csv_path):
    pdb_to_group = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_to_group[row['PDB']] = row['group']
    return pdb_to_group


def random_split(pdb_list, n_folds=3, seed=None):
    rng = np.random.RandomState(seed)
    values, inverse = np.unique(pdb_list, return_inverse=True)
    fold_assign = np.arange(len(values)) % n_folds
    rng.shuffle(fold_assign)
    return fold_assign[inverse]


def _extract_features(sample):
    return torch.cat([
        sample['wsi'].flatten(),
        sample['ws'].flatten(),
        sample['wz'],
    ], dim=0)


def _build_paired_dataset(data_mut, data_wt, group_transform=None):
    if group_transform is None:
        group_transform = lambda t: t.split('_')[2]

    x_list = []
    y_list = []
    group_list = []
    pair_index = []

    group_to_indices = {}
    visited = set()

    for i, data in enumerate([data_mut, data_wt]):
        samples, labels = data

        for j, sample in enumerate(samples):
            g = group_transform(sample['subpath'])

            if i == 0:
                group_to_indices.setdefault(g, []).append(j)
            else:
                if g not in group_to_indices or g in visited:
                    continue
                visited.add(g)
                wt_idx = len(x_list)
                for mut_idx in group_to_indices[g]:
                    pair_index.append([mut_idx, wt_idx])

            group_list.append(g)
            x_list.append(_extract_features(sample))
            y_list.append(labels[j])

    dataset = ProteinDataset(
        x=torch.stack(x_list).float(),
        y=torch.stack(y_list).float(),
        group=np.array(group_list))
    pair_index = torch.LongTensor(pair_index)

    return PairedDataset(dataset, pair_index)


def load_dataset(name):
    if name == 'skempi':
        return _build_paired_dataset(
            joblib.load('processed_data/skempi_mut.pkl'),
            joblib.load('processed_data/skempi_wt.pkl'))
    elif name == 'her2':
        her2 = joblib.load('processed_data/her2.pkl')
        return _build_paired_dataset(
            [her2[0][1:], her2[1][1:]],
            [her2[0][:1], her2[1][:1]])
    else:
        raise ValueError(f'Unknown dataset: {name}')


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, group):
        self.x = x
        self.y = y
        self.group = group

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pair_index):
        self.dataset = dataset
        self.pair_index = pair_index

    def __len__(self):
        return len(self.pair_index)

    def __getitem__(self, idx):
        mut_idx, wt_idx = self.pair_index[idx]
        return idx, self.dataset[mut_idx], self.dataset[wt_idx]

    def get_pair_groups(self, pair_indices):
        return self.dataset.group[self.pair_index[pair_indices, 0]]
