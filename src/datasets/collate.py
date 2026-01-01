import torch


def collate_fn(dataset_items: list[dict]):
    xs = [item[0] for item in dataset_items]
    return torch.stack(xs)    
  
