import numpy as np
import torch
import os
import sys

from bench2driveMMCV.datasets.B2D_vad_dataset import B2D_VAD_Dataset

from pad_config import train_pipeline,test_pipeline,modality,class_names,NameMapping,eval_cfg,point_cloud_range
import pickle
import gzip
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()  # 혹시 모를 인터랙티브 모드 차단

def dump_feature_target_to_pickle(path, data_dict) -> None:
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def compute_corners(boxes):
    # Calculate half dimensions
    x = boxes[:, 0]        # x-coordinate of the center
    y = boxes[:, 1]        # y-coordinate of the center
    half_width = boxes[:, 2]  / 2
    half_length = boxes[:, 3]  / 2
    headings= boxes[:, 4]

    cos_yaw = np.cos(headings)[...,None]
    sin_yaw = np.sin(headings)[...,None]

    # Compute the four corners
    corners_x = np.stack([half_length, -half_length, -half_length, half_length],axis=-1)
    corners_y = np.stack([half_width, half_width, -half_width, -half_width],axis=-1)

    # Rotate corners by yaw
    rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

    # Translate corners to the center of the bounding box
    corners = np.stack((rot_corners_x + x[...,None], rot_corners_y + y[...,None]), axis=-1)
    # FRONT_LEFT = 0
    # REAR_LEFT = 1
    # REAR_RIGHT = 2
    # FRONT_RIGHT = 3
    return corners


class CustomNuScenes3DDataset(B2D_VAD_Dataset):
    def __init__(self,type,ann_file, pipeline, modality):
        super().__init__(point_cloud_range=point_cloud_range,queue_length=1,data_root=data_root,ann_file=ann_file,eval_cfg=eval_cfg,map_file=map_file,pipeline=pipeline,name_mapping=NameMapping,modality=modality,classes=class_names)
        self.type=type
        self._cache_path = cache_path+type+"/"

        self.bev_pixel_width: int = 256
        self.bev_pixel_height: int = 256 // 2
        self.bev_pixel_size= 0.25

    def get_fut_box(self,gt_agent_feats,gt_agent_boxes,T=6):
        agent_num = gt_agent_feats.shape[0]

        gt_agent_fut_trajs = gt_agent_feats[..., :T * 2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T * 2:T * 3].reshape(-1, 6)
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
        gt_agent_fut_yaw = gt_agent_feats[..., T * 3 + 10:T * 4 + 10].reshape(-1, 6, 1)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:, 6:7] = -1 * (gt_agent_boxes[:, 6:7] + np.pi / 2)  # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw+ gt_agent_boxes[:, np.newaxis, 6:7]

        x=gt_agent_fut_trajs[:,:,0]
        y=gt_agent_fut_trajs[:,:,1]
        yaw=gt_agent_fut_yaw[:,:,0]

        agent_width= gt_agent_boxes[:,None,3].repeat(1,T)
        agent_length= gt_agent_boxes[:,None,4].repeat(1,T)

        fut_boxes=torch.stack([x,y,agent_width,agent_length,yaw],dim=-1)#[x, y, z, w, l, h, yaw]

        fut_boxes=fut_boxes*gt_agent_fut_mask[:,:,None]

        corners=compute_corners(fut_boxes.numpy().reshape(-1,5)).reshape(agent_num,T,4,2)

        return corners.astype(np.float32)

    def get_agents_fut_trajs_ego(self, gt_agent_feats, gt_agent_boxes, T=6, boxes_in_lidar=False):
        import numpy as np, torch
        # to numpy
        feats_np = gt_agent_feats.detach().cpu().numpy() if isinstance(gt_agent_feats, torch.Tensor) else gt_agent_feats
        boxes_np = (gt_agent_boxes.detach().cpu().numpy().copy()
                    if isinstance(gt_agent_boxes, torch.Tensor) else gt_agent_boxes.copy())

        A = feats_np.shape[0]
        fut_trajs = feats_np[..., :T*2].reshape(A, T, 2)
        fut_mask  = feats_np[..., T*2:T*3].reshape(A, T)
        fut_yaw   = feats_np[..., T*3+10:T*4+10].reshape(A, T, 1)
        fut_trajs = np.cumsum(fut_trajs, axis=1)
        fut_yaw   = np.cumsum(fut_yaw,   axis=1)

        if not boxes_in_lidar:
            boxes_np[:, 6:7] = -1.0 * (boxes_np[:, 6:7] + np.pi/2)

        fut_trajs_ego = fut_trajs + boxes_np[:, None, 0:2]
        fut_yaw_ego   = fut_yaw   + boxes_np[:, None, 6:7]

        fut_centers_ego = fut_trajs_ego.astype(np.float32)
        fut_mask_bool   = fut_mask.astype(bool)
        fut_yaw_ego     = fut_yaw_ego[..., 0].astype(np.float32)
        return fut_centers_ego, fut_mask_bool, fut_yaw_ego


    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self.bev_pixel_width / 2.0]])
        coords_idcs = (coords[:,:,::-1] / self.bev_pixel_size) + pixel_center #center 0,128

        return coords_idcs.astype(np.int32)

    def __getitem__(self, idx):

        if self.type=="train":
            data = self.prepare_train_data(idx)
        else:
            data= self.prepare_test_data(idx)

        token=str(idx)

        fut_boxes=None

        if data is None:
            return {token: fut_boxes}

        fut_valid_flag = data["fut_valid_flag"]
        gt_bboxes_3d = data['gt_bboxes_3d']
        gt_attr_labels = data['gt_attr_labels']
        ego_fut_cmd = data['ego_fut_cmd']
        img = data["img"]
        img_metas=data['img_metas']
        ego_fut_trajs=data['ego_fut_trajs']

        if self.type=="train":
            gt_bboxes_3d=gt_bboxes_3d.data
            gt_attr_labels=gt_attr_labels.data
            ego_fut_cmd=ego_fut_cmd.data
            img=img.data[0]
            img_metas=img_metas.data[0]
            ego_fut_trajs=ego_fut_trajs.data[0]
        else:
            fut_valid_flag=fut_valid_flag[0]
            gt_bboxes_3d=gt_bboxes_3d[0].data
            gt_attr_labels=gt_attr_labels[0].data
            ego_fut_cmd=ego_fut_cmd[0].data
            img=img[0].data
            img_metas=img_metas[0].data
            ego_fut_trajs=ego_fut_trajs[0].data[0]

        if fut_valid_flag :
            # gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            #     dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
            # gt_agent_feats: (B, A, 34)
            #     dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            #     lcf_feat (x, y, yaw, vx, vy, width, length, height, type)

            gt_agent_boxes=gt_bboxes_3d.tensor

            #[x, y, z, w, l, h, yaw]

            #agent_mask=(category_index!=4) & (category_index!=6)

            # SH
            fut_boxes=self.get_fut_box(gt_attr_labels,gt_agent_boxes)

            #c=gt_bboxes_3d.corners

            features = {}

            ann_info = self.data_infos[idx]

            ego_vel =ann_info["ego_vel"] [:1]#np.array([ann_info['speed'],0,0])
            ego_accel = ann_info["ego_accel"][:2]#np.linalg.norm(ann_info["ego_accel"][:2])  #np.array([ann_info['acceleration'][0],-ann_info['acceleration'][1],ann_info['acceleration'][2]])

            ego_translation = ann_info['ego_translation']

            command_near_xy = np.array(
                [ann_info['command_near_xy'][0] - ego_translation[0], ann_info['command_near_xy'][1] - ego_translation[1]])

            yaw = ann_info['ego_yaw']
            raw_theta = -(yaw-np.pi/2)
            theta_to_lidar = raw_theta

            rotation_matrix = np.array(
                [[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)], [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])
            local_command_xy = rotation_matrix @ command_near_xy

            gt_ego_fut_cmd = ego_fut_cmd.reshape(6)

            features["ego_status"] = torch.cat([ torch.tensor(ego_vel),torch.tensor(ego_accel),torch.tensor(local_command_xy), gt_ego_fut_cmd])[None].to(torch.float32)

            # origin
            # features["camera_feature"] = img[:4]

            # SH
            img_uint8 = (img[:4].clamp(0, 1) * 255).to(torch.uint8).cpu()
            features["camera_feature"] = img_uint8

            image_shape= np.zeros([1,2])

            image_shape[:,0]=img.shape[-2]
            image_shape[:,1]=img.shape[-1]

            features["img_shape"] = image_shape

            features["lidar2img"] = np.array(img_metas['lidar2img'])[:4]

            token_path = self._cache_path + str(token)

            os.makedirs(token_path, exist_ok=True)

            data_dict_path = Path(token_path) / "pad_feature.gz"

            dump_feature_target_to_pickle(data_dict_path, features)

            targets = {}

            target_traj=ego_fut_trajs.cumsum(dim=-2)

            target_traj[:,2]+=np.pi/2

            targets["trajectory"] = target_traj

            targets["token"] = token

            town_name = ann_info['town_name']
            targets["town_name"]=town_name
            world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
            targets["lidar2world"]=np.linalg.inv(world2lidar) #lidar postion

            # SH agent future trajectory
            fut_centers_ego, fut_mask_bool, fut_yaw_ego = self.get_agents_fut_trajs_ego(gt_attr_labels, gt_agent_boxes, T=6, boxes_in_lidar=True)
            if isinstance(gt_attr_labels, torch.Tensor):
                agents_category = gt_attr_labels[:, 27].detach().cpu().numpy().astype(np.int64)
            else:
                agents_category = np.asarray(gt_attr_labels[:, 27]).astype(np.int64)
            
            targets["agents_fut_centers"] = fut_centers_ego.astype(np.float32)
            targets["agents_fut_mask"]    = fut_mask_bool
            targets["agents_fut_yaw"]     = fut_yaw_ego.astype(np.float32)
            targets["agents_category"]    = agents_category 

            if self.type == "train":
                gt_bboxes_3d_bev = gt_bboxes_3d.bev  # x,y,w,l,heading
            
                #gt_bboxes_3d_bev[:, -1] = -(gt_bboxes_3d_bev[:, -1] + np.pi / 2)
            
                distances = np.linalg.norm(gt_bboxes_3d_bev[:, :2], axis=-1)
            
                gt_bboxes_3d_sort = gt_bboxes_3d_bev[np.argsort(distances)][:30]
            
                gt_bboxes_3d_all = np.concatenate([gt_bboxes_3d_sort, np.zeros([30 - len(gt_bboxes_3d_sort), 5])],
                                                  axis=0)
            
                agent_labels = np.zeros([30])
            
                agent_labels[:len(gt_bboxes_3d_sort)] = True
            
                targets["agent_states"] = gt_bboxes_3d_all
                targets["agent_labels"] = agent_labels
            
                map_gt_bboxes_3d = data['map_gt_bboxes_3d'].data.instance_list
                map_gt_labels_3d = data[
                    'map_gt_labels_3d'].data  # {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}
                bev_semantic_map = np.zeros((self.bev_pixel_height, self.bev_pixel_width),
                                            dtype=np.int64)  # 128,256  front 128
            
                for map_label in range(6):
                    map_linestring_mask = np.zeros((self.bev_pixel_height, self.bev_pixel_width)[::-1],
                                                   dtype=np.uint8)  # 256,128
                    for label, linestring in zip(map_gt_labels_3d, map_gt_bboxes_3d):
                        if label == map_label:
                            points = np.array(linestring.coords).reshape((-1, 1, 2))
                            points = self._coords_to_pixel(points)  #
                            cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
                    map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
                    entity_mask = map_linestring_mask > 0
                    bev_semantic_map[entity_mask] = map_label + 1
            
                corners = compute_corners(gt_bboxes_3d_bev)#fut_boxes[:,0]#
                category_index = gt_attr_labels[:, 27].to(int)
            
                for agent_label in range(8):
                    box_polygon_mask = np.zeros((self.bev_pixel_height, self.bev_pixel_width)[::-1], dtype=np.uint8)
                    for label, coords in zip(category_index, corners):
                        if label == agent_label:
                            exterior = coords.reshape((-1, 1, 2))
                            exterior = self._coords_to_pixel(exterior)
                            cv2.fillPoly(box_polygon_mask, [exterior], color=255)
                    box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
                    entity_mask = box_polygon_mask > 0
                    bev_semantic_map[entity_mask] = agent_label + 7
            
                targets["bev_semantic_map"] = bev_semantic_map

                # save_dir = "/mnt/mydisk/ipad_sh/debug_bev_seg"
                # os.makedirs(save_dir, exist_ok=True)

                # bev_img = (bev_semantic_map.astype(np.uint8) * 15)  # 값이 너무 작을 경우 대비 (0~14 범위 → 대략 대비 강화)
                # bev_img_color = cv2.applyColorMap(bev_img, cv2.COLORMAP_TURBO)  # 다양한 색감 선택 가능: JET, HSV, TURBO 등

                # save_path = os.path.join(save_dir, f"bev_sem_{idx:06d}.png")
                # cv2.imwrite(save_path, bev_img_color)
                # print(f"[Saved BEV semantic map] {save_path}")

                # import matplotlib
                # import matplotlib.pyplot as plt

                # # 저장 경로 지정
                # save_path_debug = f"./debug_1012_new4/vis_bev_example_{idx}.png"
                # os.makedirs(os.path.dirname(save_path_debug) or ".", exist_ok=True)

                # plt.figure(figsize=(8, 8))

                # # 1) 맵 라인
                # for label, linestring in zip(map_gt_labels_3d, map_gt_bboxes_3d):
                #     plt.plot(linestring.xy[0], linestring.xy[1], 'grey', linewidth=1.0, alpha=0.8, zorder=1)

                # # 2) 에고 미래 궤적
                # plt.plot(target_traj[:, 0], target_traj[:, 1], color='red', linewidth=2.0, alpha=0.9, zorder=3)

                # # 3) 에고 박스
                # half_length = 2.042
                # half_width  = 0.925
                # rear_axle_to_center = 0.39
                # ego_box = compute_corners(np.array([[0, rear_axle_to_center, half_width*2, half_length*2, np.pi/2]]))[0]
                # plt.plot(ego_box[:, 0], ego_box[:, 1], color='red', linewidth=1.5, zorder=3)

                # # 4) 주변 에이전트 박스 색상
                # agent_color_map = {0: 'blue', 4: 'yellow', 5: 'green'}
                # for label, agent in zip(category_index, corners):
                #     c = agent_color_map.get(int(label), 'white')
                #     plt.plot(agent[:, 0], agent[:, 1], color=c, linewidth=1.0, alpha=0.9, zorder=2)

                # # 에고 기준 미래 궤적 계산
                # fut_centers_ego, fut_mask, fut_yaw_ego = self.get_agents_fut_trajs_ego(
                #     gt_attr_labels, gt_agent_boxes, T=6, boxes_in_lidar=True  
                # )

                # # 시각화 (기존 코너 그리는 루프 뒤에)
                # A, T = fut_centers_ego.shape[:2]
                # for a in range(A):
                #     label = int(category_index[a])
                #     if label not in [0,5]:
                #         continue
                #     valid = fut_mask[a]
                #     if not np.any(valid):
                #         continue
                #     traj = fut_centers_ego[a, valid]  # (n,2), 에고 좌표계 (LiDAR 프레임)
                #     # category별 색상 선택은 기존 로직 재사용
                #     c = agent_color_map.get(int(category_index[a]), 'white')
                #     plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5, alpha=0.95, zorder=4)
                #     plt.scatter(traj[-1, 0], traj[-1, 1], s=12, zorder=5)

                # # 5) 보기 영역/스타일
                # plt.xlim(-32, 32)
                # plt.ylim(-32, 32)
                # plt.gca().set_aspect('equal', 'box')
                # plt.axis('off')

                # # === 저장 ===
                # plt.tight_layout()
                # plt.savefig(save_path_debug, dpi=300, bbox_inches='tight')
                # plt.close()
                # print(f"[Saved] {save_path_debug}")

                # for label, linestring in zip(map_gt_labels_3d, map_gt_bboxes_3d):
                #     plt.plot(linestring.xy[0],linestring.xy[1],'grey')
                
                # plt.plot(target_traj[:,0],target_traj[:,1],'red')
                
                # half_length = 2.042
                # half_width = 0.925
                # rear_axle_to_center = 0.39
                
                # ego_box=compute_corners(np.array([[0,rear_axle_to_center,half_width*2,half_length*2,np.pi/2]]))[0]
                
                # plt.plot(ego_box[:,0],ego_box[:,1],'red')
                
                # for label,agent in zip(category_index,corners):
                #     if label==0:
                #         plt.plot(agent[:,0],agent[:,1],'blue')
                #     if label==4:
                #         plt.plot(agent[:,0],agent[:,1],'yellow')
                #     if label==5:
                #         plt.plot(agent[:,0],agent[:,1],'green')
                
                # plt.xlim(-32,32)
                # plt.ylim(-32,32)
                
                # plt.show()

            data_dict_path = Path(token_path) / "pad_target.gz"

            dump_feature_target_to_pickle(data_dict_path, targets)

        return {token:fut_boxes}

def my_collate(batch):
    return batch

data_root="/data1/shyuA6000/iPad/Bench2DriveZoo/data/bench2drive"
# data_root="Bench2DriveZoo/data/bench2drive"
cache_path=os.environ["NAVSIM_EXP_ROOT"] + "/B2d_cache/"

if not os.path.exists(cache_path):
    # Create the directory
    os.makedirs(cache_path)
    print(f"Directory '{cache_path}' created.")
else:
    print(f"Directory '{cache_path}' already exists.")

for type in ['train','val']  :#
    fut_box = {}

    anno_root ="Bench2DriveZoo/data/infos/b2d_"
    map_file=anno_root+"map_infos.pkl"
    ann_file =  anno_root +"infos_"+ type+".pkl"
    if type=="train":
        pipeline=train_pipeline
    else:
        pipeline=test_pipeline
    nuscenes_data=CustomNuScenes3DDataset(type,ann_file,pipeline,modality)

    data_loader=DataLoader(nuscenes_data,batch_size=1, num_workers=28,prefetch_factor=4,pin_memory=False,collate_fn=my_collate)#

    for  data in tqdm(data_loader):
        for key,value in data[0].items():
            fut_box[key]=value

    save_path=type+"_fut_boxes.gz"

    data_dict_path=Path(cache_path)/save_path
    dump_feature_target_to_pickle(data_dict_path, fut_box)