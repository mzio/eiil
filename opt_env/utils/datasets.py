def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def get_envs(cuda=True, flags=None, train_shuffle=True, transform=None):
    assert flags is not None
    
    dataloaders = load_dataloaders(flags, train_shuffle, transform)
    envs = []
    for dix, dataloader in enumerate(dataloaders):
        targets_all = dataloader.dataset.targets_all
        if dix == 0:
            # Envs correspond to each color?
            for c in np.unique(targets_all['spurious']):
                indices_by_c = np.where(targets_all['spurious'] == c)[0]
                try:
                    dataset = get_resampled_set(dataloader.dataset, 
                                                indices_by_c, 
                                                copy_dataset=True)
                except:
                    print(len(dataloader.dataset))
                    print(len(indices_by_c))

                samples = dict()
                dataloader_ = DataLoader(dataset,
                                        batch_size=flags.bs_trn,
                                        shuffle=train_shuffle,
                                        num_workers=flags.num_workers)
                samples['dataloader'] = dataloader_
                samples['source_dataloader'] = dataloader
                samples['labels'] = dataloader.dataset.targets
                samples['indices'] = indices_by_c
        else:
            samples = dict()
            samples['dataloader'] = dataloader
            samples['labels'] = dataloader.dataset.targets
            samples['indices'] = np.arange(len(samples['labels']))
        envs.append(samples)
    return envs
