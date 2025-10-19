import os
from typing import Tuple
from pathlib import Path
import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from typing import Any, Dict, List, Union
import torch

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "b2d_training"

def collate_trim_to_min(batch: List[Any]) -> Any:
    """
    기본 default_collate가 가변 길이 첫 축(T) 때문에 실패할 때,
    배치 내 공통 최소 길이(T_min)로 잘라서 스택해주는 collate 함수.
    - 텐서/ndarray이고 첫 축만 다르고 나머지 축은 같은 경우에만 트림.
    - dict/list/tuple는 재귀 처리.
    - 스칼라는 기본 텐서 변환.
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]

    # dict 재귀
    if isinstance(elem, dict):
        out: Dict[str, Any] = {}
        for k in elem.keys():
            out[k] = collate_trim_to_min([b[k] for b in batch])
        return out

    # list/tuple 재귀
    if isinstance(elem, (list, tuple)):
        transposed = list(zip(*batch))
        return [collate_trim_to_min(list(x)) for x in transposed]

    # 텐서/넘파이 처리
    if isinstance(elem, (torch.Tensor, np.ndarray)):
        tensors = [torch.as_tensor(x) for x in batch]
        # 모두 같은 shape면 그대로 stack
        if all(t.shape == tensors[0].shape for t in tensors):
            return torch.utils.data._utils.collate.default_collate(tensors)
        # 첫 축만 다르고 나머지 축 동일하면 최소 길이로 트림 후 stack
        if all(t.dim() >= 1 for t in tensors) and all(t.shape[1:] == tensors[0].shape[1:] for t in tensors):
            tmin = min(t.shape[0] for t in tensors)
            trimmed = [t[:tmin] for t in tensors]
            return torch.stack(trimmed, dim=0)
        # 그 외엔 기본 동작 (여기서 실패할 수 있지만, 우리가 다루는 케이스는 위 분기로 커버됨)
        return torch.utils.data._utils.collate.default_collate(tensors)

    # 스칼라 등
    return torch.utils.data._utils.collate.default_collate(batch)

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=['train'],
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=['val'],
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,drop_last=True, collate_fn=collate_trim_to_min)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False,drop_last=True, collate_fn=collate_trim_to_min)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
