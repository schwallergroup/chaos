import torch

__all__ = ["torch_delete_rows"]


def torch_delete_rows(tensor, indices):
    indices_to_keep = torch.ones(tensor.shape[0], dtype=torch.bool)
    indices_to_keep[indices] = False
    return tensor[indices_to_keep]
