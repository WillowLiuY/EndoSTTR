import torch.utils.data as data
from dataset.scared import ScaredDataset

def build_data_loader(args):
    '''
    Data loader for Scared.
    
    :param args: arg parser object
    :return: data loaders for training, validation and test sets.
    '''
    if not args.dataset_directory:
        raise ValueError('Data path must be specified.')
    dataset_directory = args.dataset_directory
    
    # Dataset options
    if args.dataset.lower() == 'scared':
        train_set = ScaredDataset(dataset_directory, 'train')
        val_set = ScaredDataset(dataset_directory, 'validation')
        test_set = ScaredDataset(dataset_directory, 'test')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    # Create data loaders
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True) #pin_memory for faster GPU data loading
    
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)
    
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)

    return train_loader, val_loader, test_loader
