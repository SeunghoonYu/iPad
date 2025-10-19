from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from navsim.agents.pad.pad_model import PadModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.pad.pad_features import PadTargetBuilder
from navsim.agents.pad.pad_features import PadFeatureBuilder
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
from .score_module.compute_b2d_score import compute_corners_torch
from navsim.agents.transfuser.transfuser_loss import _agent_loss

class PadAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr: float,
            checkpoint_path: str = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data=False

        if not cache_data:
            self._pad_model = PadModel(config)

        if not cache_data and self._checkpoint_path == "":#only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                if self.b2d:
                    self.worker = RayDistributedNoTorch(threads_per_node=8)
                else:
                    self.worker = RayDistributedNoTorch(threads_per_node=16)
                self.worker_map=worker_map

            if config.b2d:
                self.train_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz")
                self.test_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/val_fut_boxes.gz")
                from .score_module.compute_b2d_score import get_scores
                self.get_scores = get_scores

                map_file =os.getenv("NAVSIM_EXP_ROOT") +"/map.pkl"

                with open(map_file, 'rb') as f:
                    self.map_infos = pickle.load(f)
                self.cuda_map=False

            else:
                from .score_module.compute_navsim_score import get_scores

                metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths

                self.get_scores = get_scores

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            self.load_state_dict({k.replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()})

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )
    
    def get_target_builders(self) :
        return [PadTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [PadFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._pad_model(features)

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        if self.b2d:
            data_points = []

            lidar2worlds=targets["lidar2world"]

            all_proposals = torch.cat([proposals, target_trajectory[:,None]], dim=1)

            all_proposals_xy=all_proposals[:, :,:, :2]
            all_proposals_heading=all_proposals[:, :,:, 2:]

            all_pos = all_proposals_xy.reshape(len(target_trajectory),-1, 2)

            mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2

            dists = torch.linalg.norm(all_pos - mid_points[:,None], dim=-1).amax(1) + 5

            xyz = torch.cat(
                [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])], dim=-1)

            xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]

            vel=torch.cat([all_proposals_xy[:, :,:1], all_proposals_xy[:,:, 1:] - all_proposals_xy[:,:, :-1]],dim=2)/ 0.5

            proposals_05 = torch.cat([all_proposals_xy + vel*0.5, all_proposals_heading], dim=-1)

            proposals_1 = torch.cat([all_proposals_xy + vel*1, all_proposals_heading], dim=-1)

            proposals_ttc = torch.stack([all_proposals, proposals_05,proposals_1], dim=3)

            ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(proposals_ttc.shape[0],proposals_ttc.shape[1], proposals_ttc.shape[2], proposals_ttc.shape[3],  4, 2)

            ego_corners_center = torch.cat([ego_corners_ttc[:,:,:,0], all_proposals_xy[:, :, :, None]], dim=-2)

            ego_corners_center_xyz = torch.cat(
                [ego_corners_center, torch.zeros_like(ego_corners_center[..., :1]), torch.ones_like(ego_corners_center[..., :1])], dim=-1)

            global_ego_corners_centers = torch.einsum("nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz)[..., :2]

            accs = torch.linalg.norm(vel[:,:, 1:] - vel[:,:, :-1], dim=-1) / 0.5

            turning_rate=torch.abs(torch.cat([all_proposals_heading[:, :,:1,0]-np.pi/2, all_proposals_heading[:,:, 1:,0]-all_proposals_heading[:,:, :-1,0]],dim=2)) / 0.5

            comforts = (accs[:,:-1] < accs[:,-1:].max()).all(-1) & (turning_rate[:,:-1] < turning_rate[:,-1:].max()).all(-1)

            if self.cuda_map==False:
                for key, value in self.map_infos.items():
                    self.map_infos[key] = torch.tensor(value).to(target_trajectory.device)
                self.cuda_map=True

            for token, town_name, proposal,target_traj, comfort, dist, xy,global_conners,local_corners in zip(targets["token"], targets["town_name"], proposals.cpu().numpy(),  target_trajectory.cpu().numpy(), comforts.cpu().numpy(), dists.cpu().numpy(), xys, global_ego_corners_centers,ego_corners_ttc.cpu().numpy()):
                all_lane_points = self.map_infos[town_name[:6]]

                # [L, 4] = (x, y, width, lane_id)
                # 장치/dtype 맞추기
                all_lane_points = all_lane_points.to(xys.device).to(xys.dtype)

                # 현재 위치(xy)로부터 반경 dist 이내의 차선만 추출
                dist_to_cur = torch.linalg.norm(all_lane_points[:, :2] - xy, dim=-1)
                nearby_point = all_lane_points[dist_to_cur < dist]

                # proposals/GT의 모양 파악
                # proposal: [P, T, 3]  (xyθ),  global_conners: [P, T, K, 2]
                P, T, K = global_conners.shape[0], global_conners.shape[1], global_conners.shape[2]

                def _empty_ego_areas():
                    # [P, T] bool 텐서들로 구성 -> 마지막에 [P, T, 3]으로 스택
                    _on_road_all = torch.zeros((P, T), dtype=torch.bool, device=global_conners.device)
                    _on_route_all = torch.zeros((P, T), dtype=torch.bool, device=global_conners.device)
                    _multi_lanes = torch.zeros((P, T), dtype=torch.bool, device=global_conners.device)
                    return torch.stack([_multi_lanes, _on_road_all, _on_route_all], dim=-1)  # [P, T, 3]

                if nearby_point.numel() == 0:
                    # 근방에 차선 자체가 없으면 안전 탈출 (offroad/비도로 씬 등)
                    ego_areas = _empty_ego_areas()
                else:
                    lane_xy   = nearby_point[:, :2]                               # [L, 2]
                    lane_width = nearby_point[:, 2]                               # [L]
                    lane_id   = nearby_point[:, -1]                               # [L]

                    # 유효 차선만 남기기: 폭>0, finite
                    valid_lane = torch.isfinite(lane_width) & (lane_width > 0)
                    lane_xy    = lane_xy[valid_lane]
                    lane_width = lane_width[valid_lane]
                    lane_id    = lane_id[valid_lane]

                    if lane_xy.shape[0] == 0:
                        ego_areas = _empty_ego_areas()
                    else:
                        # [L, P, T, K] 각 코너/센터까지의 거리
                        dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)

                        # 차선 폭 안에 들어오면 on-road
                        # (lane_width: [L] -> [L,1,1,1]로 브로드캐스트)
                        shifted = dist_to_lane - lane_width[:, None, None, None]

                        # L이 0이 아닌 게 확정이므로 .min(dim=0)로 안전하게 argmin/값 동시 계산
                        dist_min, nearest_lane = shifted.min(dim=0)               # [P, T, K] 각각의 최근접 차선 idx
                        on_road = (dist_to_lane < lane_width[:, None, None, None])# [L, P, T, K] bool
                        on_road_all = on_road.any(0).all(-1)                      # [P, T] 모든 코너가 어떤 차선 안에 있는가

                        # 최근접 차선 id 맵
                        # nearest_lane: [P, T, K] -> lane_id[nearest_lane]: 같은 shape
                        nearest_lane_id = lane_id[nearest_lane]                   # [P, T, K]

                        # 마지막 K(센터) 인덱스가 center임을 가정하고 경로 id 취득
                        center_nearest_lane_id = nearest_lane_id[:, :, -1]        # [P, T]
                        nearest_road_id = torch.round(center_nearest_lane_id)     # [P, T]

                        # GT(마지막 P-1이 GT이면 아래처럼 사용, 아니라면 원 코드의 의도대로 유지)
                        target_road_id = torch.unique(nearest_road_id[-1])        # [*] 이 샘플의 GT가 지나간 도로 id set
                        on_route_all = torch.isin(nearest_road_id, target_road_id) # [P, T]

                        # 여러 차선을 동시에 밟는가: 코너 기준으로 lane id가 섞이면 True
                        corner_nearest_lane_id = nearest_lane_id[:, :, :-1]       # [P, T, K-1] (코너들)
                        batch_multiple_lanes_mask = (corner_nearest_lane_id != corner_nearest_lane_id[:, :, :1]).any(-1)  # [P, T]

                        # on_road_all을 GT on/off-road 패턴과 정렬(원 코드 유지)
                        on_road_all = (on_road_all == on_road_all[-1:])

                        ego_areas = torch.stack([batch_multiple_lanes_mask, on_road_all, on_route_all], dim=-1)  # [P, T, 3]

                data_dict = {
                    "fut_box_corners": metric_cache_paths[token],
                    "_ego_coords": local_corners,
                    "target_traj": target_traj,
                    "proposal": proposal,
                    "comfort": comfort,
                    "ego_areas": ego_areas.detach().to('cpu', non_blocking=True).numpy(),
                }
                data_points.append(data_dict)

                
                # all_lane_points = self.map_infos[town_name[:6]]

                # dist_to_cur = torch.linalg.norm(all_lane_points[:,:2] - xy, dim=-1)

                # nearby_point = all_lane_points[dist_to_cur < dist]

                # lane_xy = nearby_point[:, :2]
                # lane_width = nearby_point[:, 2]
                # lane_id = nearby_point[:, -1]

                # dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)

                # on_road = dist_to_lane < lane_width[:, None, None, None]

                # on_road_all = on_road.any(0).all(-1)

                # nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None,None], dim=0)

                # nearest_lane_id=lane_id[nearest_lane]

                # center_nearest_lane_id=nearest_lane_id[:,:,-1]

                # nearest_road_id = torch.round(center_nearest_lane_id)

                # target_road_id = torch.unique(nearest_road_id[-1])

                # on_route_all = torch.isin(nearest_road_id, target_road_id)
                # # in_multiple_lanes: if
                # # - more than one drivable polygon contains at least one corner
                # # - no polygon contains all corners
                # corner_nearest_lane_id=nearest_lane_id[:,:,:-1]

                # batch_multiple_lanes_mask = (corner_nearest_lane_id!=corner_nearest_lane_id[:,:,:1]).any(-1)

                # on_road_all=on_road_all==on_road_all[-1:]
                # # on_road_all = on_road_all | ~on_road_all[-1:]# on road or groundtruth offroad

                # ego_areas=torch.stack([batch_multiple_lanes_mask,on_road_all,on_route_all],dim=-1)

                # data_dict = {
                #     "fut_box_corners": metric_cache_paths[token],
                #     "_ego_coords": local_corners,
                #     "target_traj": target_traj,
                #     "proposal":proposal,
                #     "comfort": comfort,
                #     "ego_areas": ego_areas.cpu().numpy(),
                # }
                # data_points.append(data_dict)
        else:
            data_points = [
                {
                    "token": metric_cache_paths[token],
                    "poses": poses,
                    "test": test
                }
                for token, poses in zip(targets["token"], proposals.cpu().numpy())
            ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def score_loss(self, pred_logit, pred_logit2,agents_state, pred_area_logits, target_scores, gt_states, gt_valid,
                   gt_ego_areas):

        if agents_state is not None:
            pred_states = agents_state[..., :-1].reshape(gt_states.shape)
            pred_logits = agents_state[..., -1:].reshape(gt_valid.shape)

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            pred_area_loss = 0

        sub_score_loss = self.bce_logit_loss(pred_logit, target_scores[..., -pred_logit.shape[-1]:])  # .mean()[..., -6:]

        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            sub_score_loss2 = self.bce_logit_loss(pred_logit2, target_scores)  # .mean()[..., -6:-1][..., -6:-1]

            final_score_loss2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            sub_score_loss=(sub_score_loss+sub_score_loss2)/2

            final_score_loss=(final_score_loss+final_score_loss2)/2

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        #dist[dist==0]=10000

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss

    def pad_loss(self,targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config  ):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
            targets, proposals, test=False)

        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []

        for proposals_i in proposal_list:

            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(
                1).mean()

            inter_loss = self.diversity_loss(proposals_i)

            trajectory_loss = config.prev_weight * trajectory_loss  + min_loss+ inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]

        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas)
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(targets, pred, config)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
                config.trajectory_weight * trajectory_loss
                + config.sub_score_weight * sub_score_loss
                + config.final_score_weight * final_score_loss
                + config.pred_ce_weight * pred_ce_loss
                + config.pred_l1_weight * pred_l1_loss
                + config.pred_area_weight * pred_area_loss
                + config.agent_class_weight * agent_class_loss
                + config.agent_box_weight * agent_box_loss
                + config.bev_semantic_weight * bev_semantic_loss

        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            "inter_loss0": inter_loss0,
            # "inter_loss1": inter_loss1,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            # "min_loss1": min_loss1,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config)

    def get_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)#,weight_decay= 1e-2
        # self.lr_warmup_steps=500
        # self.lr_total_steps=20000
        # self.lr_min_ratio=1e-3

        # def lr_lambda(current_step):
        #     if current_step < self.lr_warmup_steps:
        #         return (1.0 / 3) + (current_step / self.lr_warmup_steps) * (1 - 1.0 / 3)
        #     return 1.0
        #     # return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
        #     #     1.0
        #     #     + math.cos(
        #     #         math.pi
        #     #         * min(
        #     #             1.0,
        #     #             (current_step - self.lr_warmup_steps)
        #     #             / (self.lr_total_steps - self.lr_warmup_steps),
        #     #         )
        #     #     )
        #     # )


        # scheduler = {
        #     'scheduler': LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',  # Update every step
        #     'frequency': 1
        # }

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return torch.optim.Adam(self._pad_model.parameters(), lr=self._lr)#,weight_decay=1e-4

    def get_training_callbacks(self):

        checkpoint_cb = ModelCheckpoint(save_top_k=20,
                                        monitor='val/score_epoch',
                                        filename='{epoch}-{step}',
                                        mode="max"
                                        )

        return [checkpoint_cb]