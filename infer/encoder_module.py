import sys
import os
import glob

import json
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.preprocessing import StandardScaler
import hdbscan
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import cv2
from PIL import Image
import matplotlib.pyplot as plt


from subpackage.clip_img_embedder import FrozenOpenCLIPImageEmbedder as FrozenOpenCLIPImageEmbedder

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

import random

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


@torch.no_grad()
def key_frame_select(args, clip_img_encoder,seq_path, yuv_w=512, yuv_h=320, n_key_frames=5):
    cos_los = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #cos_los = cosine_similarity
    seq_name = os.path.basename(seq_path)
    frm_path = os.path.join(seq_path, '*png')
    img_paths = sorted(glob.glob(frm_path, recursive=True))

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((yuv_h, yuv_w)),  
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
            )   
    ])

    toTensor = transforms.Compose([
        transforms.ToTensor()  # 转换为Tensor
    ])

    frames = len(img_paths)
    print("frames num: {}".format(frames))
    # 第一帧和最后一帧为关键帧
    n_clips = int((frames) / (n_key_frames - 1))
    print("clips len: {}".format(n_clips))

    first_frame_path = img_paths[0]
    first_rgb_ori = Image.open(first_frame_path).convert("RGB")

    first_rgb = transform(first_rgb_ori).unsqueeze(0).to(DEFAULT_DEVICE)
    first_image_embeddings = clip_img_encoder(first_rgb).flatten().unsqueeze(0)


    d_lis = [1]  # 记录相似度
    frames_index = [1]
    key_frame_index = [1]  # 关键帧索引

    key_frame_rgb = first_rgb_ori # 关键帧
    key_frame_rgb_lis = [key_frame_rgb]  
    # frame_rgb_lis_ori = [first_rgb_ori]

    num = 1
    d_min = -1
    key_frame_idx = 1
    key_frame_embeddings = first_image_embeddings
    rgb_last_embeddings = first_image_embeddings
    rgb_last = key_frame_rgb
    scene_change = 0
    for img_path in img_paths[1:]:
        img_name = os.path.basename(img_path)
        num += 1

        cur_rgb_ori = Image.open(img_path).convert("RGB")
        rgb_cur = transform(cur_rgb_ori).unsqueeze(0).to(DEFAULT_DEVICE)
        rgb_embeddings = clip_img_encoder(rgb_cur).flatten().unsqueeze(0)

        d = cos_los(first_image_embeddings, rgb_embeddings)
        d_last = cos_los(rgb_last_embeddings, rgb_embeddings)

        key_fresh = 0
        
        # 场景切换 两帧都为关键帧
        if d_last < 0.5:
            key_frame_idx = num
            key_frame_rgb = cur_rgb_ori
            key_frame_embeddings = rgb_embeddings
            key_fresh = 1
            scene_change = 1

            key_frame_index.append(num-1)
            key_frame_index.append(num)

            key_frame_rgb_lis.append(rgb_last)
            key_frame_rgb_lis.append(key_frame_rgb)

            d_min = -1
            first_image_embeddings = key_frame_embeddings

        rgb_last_embeddings = rgb_embeddings
        rgb_last = cur_rgb_ori      
        #print(d.shape)
        
        if (d_min == -1 or (d < d_min and num - key_frame_index[-1] + 1 <= 30)) and key_fresh == 0:
            d_min = d
            key_frame_idx = num
            key_frame_rgb = cur_rgb_ori
            key_frame_embeddings = rgb_embeddings
            key_fresh = 1
        if num - key_frame_index[-1] + 1 == 30 and key_fresh == 0:
            d_min = d
            key_frame_idx = num
            key_frame_rgb = cur_rgb_ori
            key_frame_embeddings = rgb_embeddings
            key_fresh = 1
        if key_frame_index[-1] > 66 and num > 90 and key_fresh == 1:
            key_frame_idx = 96
        elif  key_frame_index[-1] > 60 and num > 84 and key_fresh == 1:
            key_frame_idx = 90
        
        d_lis.append(d)
        frames_index.append(num)

        if num % n_clips == 0:
            if scene_change == 0:
                key_frame_index.append(key_frame_idx)
                key_frame_rgb_lis.append(key_frame_rgb)
                d_min = -1
                first_image_embeddings = key_frame_embeddings
            else:
                scene_change = 0

        if num == 96:
            d_min = d
            key_frame_idx = num
            key_frame_rgb = cur_rgb_ori
            key_frame_embeddings = rgb_embeddings

            key_frame_index.append(key_frame_idx)
            key_frame_rgb_lis.append(key_frame_rgb)
            d_min = -1
            first_image_embeddings = key_frame_embeddings


    key_frame_index = np.unique(np.array(key_frame_index))
    print("key frames index: {}".format(key_frame_index))
    # plt.plot(torch.tensor(frames_index), torch.tensor(d_lis), 'b-', label='sematic similarity')
    # plt.scatter(torch.tensor(key_frame_index), torch.tensor([d_lis[i-1] for i in key_frame_index]), color='red', label='keyframes')
    # plt.title('similarity curve')
    # plt.xlabel('Frame idx')
    # plt.ylabel('cosine similarity')
    # plt.legend()
    # plt.savefig(f"/key_frames/{seq_name}.png")
    # plt.close()
    if not os.path.exists(f"{args.keyframes_path}"):
        os.mkdir(f"{args.keyframes_path}/")
    np.save(f"/{args.keyframes_path}{seq_name}.npy",key_frame_index)

    return key_frame_index


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="co-tracker/checkpoints/baseline_online.pth",
        # default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=64, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        default=True,
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    parser.add_argument("--data_config", type=str,default="./dataset_config_motion.json", help="path to test config")
    parser.add_argument("--keyframes", action='store_true', default=False, help="select keyframes")
    parser.add_argument("--tracker", action='store_true', default=True, help="tracker")
    parser.add_argument("--keyframes_path", default="/key_frames_24/", help="select keyframes")
    parser.add_argument("--traj_path", default="/traj_save_24_sparse/", help="select traj")

    args = parser.parse_args()
    with open(args.data_config) as f:
        config = json.load(f)
    data_root = config["root_path"]
    config = config['test_datas']
    if args.keyframes or args.tracker:
        clip_img_encoder = FrozenOpenCLIPImageEmbedder().to(DEFAULT_DEVICE)
        
    if args.checkpoint is not None:
            if args.use_v2_model:
                model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
            else:
                if args.offline:
                    window_len = 60
                else:
                    window_len = 16
                model = CoTrackerPredictor(
                    checkpoint=args.checkpoint,
                    v2=args.use_v2_model,
                    offline=args.offline,
                    window_len=window_len,
                )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)

        # device = DEFAULT_DEVICE
        # model = torch.hub.load("facebookresearch/NeuralCompression", index)
        # model = model.to(device)
        # model = model.eval()
        # model.update()
        # model.update_tensor_devices("compress")

    toTensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
            )   
    ])

    #totensor = ToTensor()
    
    cos_los = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    height = 320
    width = 512
    
    if not os.path.exists(f"{args.traj_path}"):
        os.mkdir(f"{args.traj_path}")
    for ds_name in config:
        print(ds_name)
        for seq in config[ds_name]['sequences']:
            # HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=10,    # 最小簇大小
                min_samples=6,         # 核心点所需邻居数
                cluster_selection_epsilon=0.05  # 合并相近簇的阈值
            )

            # key_frames_index = [0,15,30,45,60,75,90,95]
            print(seq)
            seq_path = data_root + ds_name + "/" + seq
            trj_path = args.traj_path + ds_name + "/" + seq + "/"
            if args.keyframes:
                key_frames_index = key_frame_select(args, clip_img_encoder,seq_path)
            else:
                seq_name = os.path.basename(seq_path)
                key_frames_index = np.load(f"/{args.keyframes_path}{seq_name}.npy")
            if not os.path.exists(args.traj_path + ds_name + "/"):
                os.mkdir(args.traj_path + ds_name + "/")            
            if not os.path.exists(f"{trj_path}"):
                os.mkdir(f"{trj_path}")
            if args.tracker:
                clips_num = len(key_frames_index) - 1
                clips = 0
                code_last = 0
                key_first = 0
                print(key_frames_index)
                for key_id in range(len(key_frames_index) - 1):
                    clips += 1
                    if not os.path.exists(f"{trj_path}/clip_{clips}"):
                        os.mkdir(f"{trj_path}/clip_{clips}")
                    if not os.path.exists(f"{trj_path}/k_means_clip_{clips}"):
                        os.mkdir(f"{trj_path}/k_means_clip_{clips}")
                    
                    #key_id = key_first 
                    # key_img_path1 = seq_path + f"/im{key_frames_index[key_id]:05}.png"
                    # key_img_path2 = seq_path + f"/im{key_frames_index[key_id+1]:05}.png"
                    # with open(key_img_path1, "rb") as f:
                    #     image_pil = Image.open(f)
                    #     image_pil = image_pil.convert("RGB")
                    # if key_id == key_first:
                    #     code_last = totensor(image_pil).unsqueeze(0).to(device) # mage_model.decompress(compress_list[num], force_cpu=False).clamp(0.0, 1.0)
                    # key_img1 = code_last #totensor(image_pil).unsqueeze(0).to(device)

                    # with open(key_img_path2, "rb") as f:
                    #     image_pil = Image.open(f)
                    #     image_pil = image_pil.convert("RGB")
                    # code_last = totensor(image_pil).unsqueeze(0).to(device) # image_model.decompress(compress_list[num+1], force_cpu=False).clamp(0.0, 1.0)
                    # key_img2 = code_last #image_model.decompress(compress_list[num+1], force_cpu=False).clamp(0.0, 1.0)#totensor(image_pil).unsqueeze(0).to(device)              
                    
                    video_length = key_frames_index[key_id+1] - key_frames_index[key_id] + 1
                    if video_length > 2:
                        video = torch.zeros((1,video_length,3,height,width))
                        video_tensor = torch.zeros((1,video_length,3,height,width))
                        for i in range(video_length):
                            ori_img_path = seq_path + f"/im{key_frames_index[key_id]+i:05}.png"
                            ori_img = np.array(Image.open(ori_img_path).convert("RGB"))
                            video[:,i,:,:,:] = torch.from_numpy(ori_img).permute(2, 0, 1).float()
                            video_tensor[:,i,:,:,:] = toTensor(ori_img)#.permute(2, 0, 1)

                        video_tensor = video_tensor.to(DEFAULT_DEVICE)
                        video_inverse = torch.flip(video_tensor, dims=[1])
                        pred_tracks, pred_visibility = model(

                            video_tensor[:,:],
                            grid_size=args.grid_size,
                            grid_query_frame=0,
                            backward_tracking=args.backward_tracking,
                        )

                        pred_tracks_inverse, pred_visibility_inverse = model(
                            video_inverse[:,:],
                            grid_size=args.grid_size,
                            grid_query_frame=0,
                            backward_tracking=args.backward_tracking,
                        )

                        pred_tracks = torch.concatenate([pred_tracks, torch.flip(pred_tracks_inverse, dims=[1])],dim=2)
                        pred_visibility = torch.concatenate([pred_visibility, torch.flip(pred_visibility_inverse, dims=[1])],dim=2)
                        
                        print("computed")

                        data = np.squeeze(pred_tracks.cpu().numpy(), axis=0)
                        data = np.transpose(data, (1, 0, 2))   
                        data_visit = np.squeeze(pred_visibility.cpu().numpy(), axis=0) 
                        data_visit = np.transpose(data_visit, (1, 0))

                        grid_len_w = int(width / args.grid_size)
                        grid_len_h = int(width / args.grid_size)

                        n_trajectories = data.shape[0]
                        features = []

                        data_new=[]
                        data_visit_new = []

                        for i in range(n_trajectories):
                            traj = data[i]  # 形状 (16, 2)
                            data_new.append(data[i])
                            data_visit_new.append(data_visit[i])

                            start = traj[0]
                            end = traj[-1]
                            dx = end[0] - start[0]
                            dy = end[1] - start[1]

                            total_dist = 0.0
                            velocities = []
                            f_list = []   
                            for t in range(video_length - 1):
                                f_list.append(traj[t+1][0])
                                f_list.append(traj[t+1][1])
                                delta = traj[t+1] - traj[t]
                                total_dist += np.linalg.norm(delta)
                                velocities.append(delta)
    
                            velocities = np.array(velocities)

                            speed_x = dx / float(video_length - 1)
                            speed_y = dy / float(video_length - 1)
                            var_x = np.var(velocities[:, 0])
                            var_y = np.var(velocities[:, 1])
                            
                            dir_changes = 0.0
                            count = 0
                            for j in range(len(velocities)-1):
                                v1 = velocities[j]
                                v2 = velocities[j+1]
                                norm_v1 = np.linalg.norm(v1)
                                norm_v2 = np.linalg.norm(v2)
                                if norm_v1 == 0 or norm_v2 == 0:
                                    continue
                                cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                                angle = np.arccos(cos_theta)
                                dir_changes += angle
                                count += 1
                            avg_dir_change = dir_changes / count if count > 0 else 0.0

                            features.append([
                                start[0], start[1],   # 起始点坐标
                                end[0], end[1],
                                traj[int(video_length // 2)][0],traj[int(video_length // 2)][1],
                                dx, dy,               # 位移向量
                                total_dist,           # 轨迹总长度
                                speed_x, speed_y,     # 平均速度
                                var_x, var_y,         # 速度方差
                                avg_dir_change        # 平均方向变化
                            ])            

                        data = np.array(data_new)
                        data_visit = np.array(data_visit_new)
                        features = np.array(features)

                        # 特征标准化
                        scaler = StandardScaler()
                        scaled_features = scaler.fit_transform(features)
                        # HDBSCAN
                        labels = clusterer.fit_predict(scaled_features)

                        n_clusters = len(np.unique(labels))  - (1 if -1 in labels else 0)  # 排除噪声点

                        mask = torch.ones((n_clusters, video_length, height, width))
                        mask_visit = torch.ones((n_clusters, video_length, height, width))
                        #trj_num = 2
                        #sementic_list = []
                        #sementic_thred = 0
                        
                        for idx in range(0, n_clusters):
                            traj_central = data[labels == idx] # (n, 16, 2)
                            traj_visit = data_visit[labels == idx] # （n,16）

                            total_dist = 0.0
                            num_visit = 0
    
                            for fr in range(video_length):
                                for trj in range(traj_central.shape[0]):
                                    x = int(traj_central[trj,fr,0])
                                    y = int(traj_central[trj,fr,1])

                                    if fr > 0 and traj_visit[trj,fr] == 1 and traj_visit[trj,fr-1] == 1:
                                        num_visit += 1
                                        delta = traj_central[trj,fr] - traj_central[trj,fr-1]
                                        total_dist += np.linalg.norm(delta)

                                    if 0 <= x < width and 0 <= y < height:
                                        mask_visit[idx,fr,max(y-grid_len_h, 0):min(y+grid_len_h,height), max(x-grid_len_w, 0 ):min(x+grid_len_w, width)] = 0     
                                    if 0 <= x < width and 0 <= y < height and traj_visit[trj,fr] == 1:
                                        mask[idx,fr,max(y-grid_len_h, 0):min(y+grid_len_h,height), max(x-grid_len_w, 0 ):min(x+grid_len_w, width)] = 0

                                if torch.sum(mask_visit[idx,fr,:,:]) < width * height * 0.6:
                                    mask[idx,fr,:,:] = 1
                                    # video

                            loss1 = 0
                            num_visit = max(num_visit,1)
                            traj_len = total_dist/ num_visit / video_length * 16 #np.sum(traj_features[:,6]) / traj_features.shape[0] / (width+height) * 2

                            for fr in range(0,video_length):
                                fr_1 = transform(video_tensor[:1,fr,:,:,:])

                                kernel = np.ones((3, 3), np.uint8)
                                # 应用膨胀操作
                                dilated_mask= cv2.dilate(((-1) *(mask[idx, fr] - 1)).numpy(), kernel, iterations=1)

                                current_mask = dilated_mask
                                current_mask = (current_mask > 0.5).astype(np.uint8) * 255


                                inpainted_frame = cv2.inpaint(video[0,fr,:,:,:].cpu().permute(1,2,0).numpy().astype(np.uint8), current_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
      
                                fr_1_modified = toTensor(inpainted_frame).unsqueeze(0).cuda()

                                img_emb1 = clip_img_encoder(fr_1)#.flatten().unsqueeze(0)
                                img_emb2 = clip_img_encoder(transform(fr_1_modified))#.flatten().unsqueeze(0)

                                los1 = 1 - cos_los(img_emb1, img_emb2)
                                loss1 += los1         

                            loss1 /= video_length / 16
                            
                            inter_matter = loss1[0]
                            sematic_matter = inter_matter * traj_len
    
                            cluster_centers = torch.zeros((2,video_length,2))
                            max_k = 15
                            delta_k = int(traj_central.shape[0] / 64 / 64 / 4 * max_k + 0.5)

                            k = max(min(int(max_k * inter_matter * 3 / 4 + delta_k + 0.5), max_k),1)#random.randint(1, max_k)
                
                            print(f"k: {k}")
                            if sematic_matter > 0.3 and traj_len > 3:    
                                data_scaled = np.dstack((traj_central, traj_visit))
         
                                model_kmeans = TimeSeriesKMeans(
                                    n_clusters=k,
                                    metric="euclidean",
                                    max_iter=50
                                )
                                # 训练模型并获取聚类标签

                                labels_kmeans = model_kmeans.fit_predict(data_scaled)
                                # # 提取聚类中心轨迹(k, 16, 2)
                                result = torch.from_numpy(model_kmeans.cluster_centers_)
                                cluster_centers = result[:,:,:2]

                                cluster_centers_visit = result[:,:,2].numpy()
 
                                for trj in range(cluster_centers.shape[0]):
                                    for fr in range(cluster_centers.shape[1]):
                                        if cluster_centers_visit[trj][fr] == 0:
                                            cluster_centers[trj][fr][0] = -1
                                            cluster_centers[trj][fr][1] = -1
                                                                          
                                for trj in range(traj_central.shape[0]):
                                    for fr in range(traj_central.shape[1]):
                                        if traj_visit[trj][fr] == 0:
                                            traj_central[trj][fr][0] = -1
                                            traj_central[trj][fr][1] = -1

                                
                                traj_matter = cluster_centers

                                np.save(f'{trj_path}/clip_{clips}/traj_{idx+1}.npy', traj_central)
                                np.save(f'{trj_path}/k_means_clip_{clips}/traj_{idx+1}.npy', traj_matter.numpy())
                                print(f"save {idx + 1}.")

    return

if __name__ == "__main__":
    main()