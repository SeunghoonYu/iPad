from typing import Any, Dict, List
import torch
import numpy as np

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def _pad_stack(tensors: List[torch.Tensor], pad_value: float = 0.0):
    # 첫 축(길이)만 다르고 나머지 축은 동일하다고 가정: [T, ...]
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    max_len = int(lengths.max().item())
    if max_len == 0:
        # 전부 길이 0인 특수 케이스
        tail_shape = tensors[0].shape[1:]
        out = torch.empty((len(tensors), 0, *tail_shape), dtype=tensors[0].dtype, device=tensors[0].device)
        mask = torch.zeros((len(tensors), 0), dtype=torch.bool, device=tensors[0].device)
        return out, lengths, mask

    tail_shape = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    device = tensors[0].device
    out = torch.full((len(tensors), max_len, *tail_shape), fill_value=pad_value, dtype=dtype, device=device)
    mask = torch.zeros((len(tensors), max_len), dtype=torch.bool, device=device)  # True=valid

    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L] = t
        mask[i, :L] = True
    return out, lengths, mask

def collate_with_padding(batch: List[Any], pad_value: float = 0.0):
    # dict 기반으로 재귀 처리
    elem = batch[0]
    if isinstance(elem, dict):
        out: Dict[str, Any] = {}
        keys = elem.keys()
        for k in keys:
            vals = [b[k] for b in batch]
            out[k] = collate_with_padding(vals, pad_value)
        return out

    # 텐서/넘파이
    if isinstance(elem, (torch.Tensor, np.ndarray)):
        tensors = [ _to_tensor(v) for v in batch ]
        # 모두 동일 shape이면 그냥 stack
        same = all(t.shape == tensors[0].shape for t in tensors)
        if same:
            return torch.stack(tensors, dim=0)
        # 첫 축만 다르고 나머지는 동일하면 pad
        if all(t.dim() >= 1 for t in tensors) and all(t.shape[1:] == tensors[0].shape[1:] for t in tensors):
            padded, lengths, mask = _pad_stack(tensors, pad_value=pad_value)
            return {"padded": padded, "lengths": lengths, "mask": mask}
        # 그 외엔 리스트로 넘겨 default_collate가 다시 시도하지 않도록 그대로 반환
        return tensors

    # 리스트/튜플
    if isinstance(elem, (list, tuple)):
        # 리스트 안 원소가 텐서/넘파이면 위 로직이 적용되도록 재귀
        transposed = list(zip(*batch))
        return [collate_with_padding(list(x), pad_value) for x in transposed]

    # 스칼라 등은 기본 텐서 변환
    return torch.tensor(batch)
