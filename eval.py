import os
import pyviz3d.visualizer as vis
from pprint import pprint

from sklearn.cluster import DBSCAN
import logging
from pytorch_lightning import Trainer, seed_everything
from omegaconf import DictConfig
import hydra
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensorQuantizationMode
import random
import colorsys
from typing import List, Tuple
import functools
import numpy as np
TestMode = 'test'
import torch
from models.mask3d import Mask3D
from trainer.trainer import InstanceSegmentation
from utils.utils import load_checkpoint_with_missing_or_exsessive_keys
"""
'exclude2',
'hook_turn',
'exclude',
'hook_pull',
'key_press',
'rotate',
'foot_push',
'unplug',
'plug_in',
'pinch_pull',
   'tip_push'
"""
SCANNET_COLOR_MAP_200 = {
    0: (152.0, 223.0, 138.0),
    1: (174.0, 199.0, 232.0),
    2: (255,255,255),
    3: (0.0, 0.0, 0.0),
    4: (255.0, 152.0, 150.0),
    5: (214.0, 39.0, 40.0),
    6: (91.0, 135.0, 229.0),
    7: (31.0, 119.0, 180.0),
    8: (229.0, 91.0, 104.0),
    9: (247.0, 182.0, 210.0),
    10: (91.0, 229.0, 110.0),
}
    #      0: (220, 20, 60),   # Crimson
    # 1: (0, 191, 255),   # Deep Sky Blue
    # 2: (50, 205, 50),   # Lime Green
    # 3: (218, 165, 32),  # Goldenrod
    # 4: (147, 112, 219), # Medium Purple
    # 5: (255, 99, 71),   # Tomato
    # 6: (47, 79, 79),    # Dark Slate Gray
    # 7: (160, 82, 45),   # Sienna
    # 8: (64, 224, 208),  # Turquoise
    # 9: (218, 112, 214), # Orchid
    # 10: (70, 130, 180),  # Steel Blue
    # 11: (255.0, 187.0, 120.0),
    # 13: (141.0, 91.0, 229.0),
    # 14: (112.0, 128.0, 144.0),
    # 15: (196.0, 156.0, 148.0),
    # 16: (197.0, 176.0, 213.0),
    # 17: (44.0, 160.0, 44.0),
    # 18: (148.0, 103.0, 189.0),
    # 19: (229.0, 91.0, 223.0),
    # 21: (219.0, 219.0, 141.0),
    # 22: (192.0, 229.0, 91.0),
    # 23: (88.0, 218.0, 137.0),
    # 24: (58.0, 98.0, 137.0),
    # 26: (177.0, 82.0, 239.0),
    # 27: (255.0, 127.0, 14.0),
    # 28: (237.0, 204.0, 37.0),
    # 29: (41.0, 206.0, 32.0),
    # 31: (62.0, 143.0, 148.0),
    # 32: (34.0, 14.0, 130.0),
    # 33: (143.0, 45.0, 115.0),
    # 34: (137.0, 63.0, 14.0),
    # 35: (23.0, 190.0, 207.0),
    # 36: (16.0, 212.0, 139.0),
    # 38: (90.0, 119.0, 201.0),
    # 39: (125.0, 30.0, 141.0),
    # 40: (150.0, 53.0, 56.0),
    # 41: (186.0, 197.0, 62.0),
    # 42: (227.0, 119.0, 194.0),
    # 44: (38.0, 100.0, 128.0),
    # 45: (120.0, 31.0, 243.0),
    # 46: (154.0, 59.0, 103.0),
    # 47: (169.0, 137.0, 78.0),
    # 48: (143.0, 245.0, 111.0),
    # 49: (37.0, 230.0, 205.0),
    # 50: (14.0, 16.0, 155.0),
    # 51: (196.0, 51.0, 182.0),
    # 52: (237.0, 80.0, 38.0),
    # 54: (138.0, 175.0, 62.0),
    # 55: (158.0, 218.0, 229.0),
    # 56: (38.0, 96.0, 167.0),
    # 57: (190.0, 77.0, 246.0),
    # 58: (208.0, 49.0, 84.0),
    # 59: (208.0, 193.0, 72.0),
    # 62: (55.0, 220.0, 57.0),
    # 63: (10.0, 125.0, 140.0),
    # 64: (76.0, 38.0, 202.0),
    # 65: (191.0, 28.0, 135.0),
    # 66: (211.0, 120.0, 42.0),
    # 67: (118.0, 174.0, 76.0),
    # 68: (17.0, 242.0, 171.0),
    # 69: (20.0, 65.0, 247.0),
    # 70: (208.0, 61.0, 222.0),
    # 71: (162.0, 62.0, 60.0),
    # 72: (210.0, 235.0, 62.0),
    # 73: (45.0, 152.0, 72.0),
    # 74: (35.0, 107.0, 149.0),
    # 75: (160.0, 89.0, 237.0),
    # 76: (227.0, 56.0, 125.0),
    # 77: (169.0, 143.0, 81.0),
    # 78: (42.0, 143.0, 20.0),
    # 79: (25.0, 160.0, 151.0),
    # 80: (82.0, 75.0, 227.0),
    # 82: (253.0, 59.0, 222.0),
    # 84: (240.0, 130.0, 89.0),
    # 86: (123.0, 172.0, 47.0),
    # 87: (71.0, 194.0, 133.0),
    # 88: (24.0, 94.0, 205.0),
    # 89: (134.0, 16.0, 179.0),
    # 90: (159.0, 32.0, 52.0),
    # 93: (213.0, 208.0, 88.0),
    # 95: (64.0, 158.0, 70.0),
    # 96: (18.0, 163.0, 194.0),
    # 97: (65.0, 29.0, 153.0),
    # 98: (177.0, 10.0, 109.0),
    # 99: (152.0, 83.0, 7.0),
    # 100: (83.0, 175.0, 30.0),
    # 101: (18.0, 199.0, 153.0),
    # 102: (61.0, 81.0, 208.0),
    # 103: (213.0, 85.0, 216.0),
    # 104: (170.0, 53.0, 42.0),
    # 105: (161.0, 192.0, 38.0),
    # 106: (23.0, 241.0, 91.0),
    # 107: (12.0, 103.0, 170.0),
    # 110: (151.0, 41.0, 245.0),
    # 112: (133.0, 51.0, 80.0),
    # 115: (184.0, 162.0, 91.0),
    # 116: (50.0, 138.0, 38.0),
    # 118: (31.0, 237.0, 236.0),
    # 120: (39.0, 19.0, 208.0),
    # 121: (223.0, 27.0, 180.0),
    # 122: (254.0, 141.0, 85.0),
    # 125: (97.0, 144.0, 39.0),
    # 128: (106.0, 231.0, 176.0),
    # 126: (12.0, 61.0, 162.0),
    # 123: (124.0, 66.0, 140.0),
    # 129: (137.0, 66.0, 73.0),
    # 12: (250.0, 253.0, 26.0),
    # 20: (55.0, 191.0, 73.0),
    # 25: (60.0, 126.0, 146.0),
    # 30: (153.0, 108.0, 234.0),
    # 37: (184.0, 58.0, 125.0),
    # 43: (135.0, 84.0, 14.0),
    # 53: (139.0, 248.0, 91.0),
    # 60: (53.0, 200.0, 172.0),
    # 61: (63.0, 69.0, 134.0),
    # 91: (190.0, 75.0, 186.0),
    # 92: (127.0, 63.0, 52.0),
    # 94: (141.0, 182.0, 25.0),
    # 108: (56.0, 144.0, 89.0),
    # 109: (64.0, 160.0, 250.0),
    # 111: (182.0, 86.0, 245.0),
    # 113: (139.0, 18.0, 53.0),
    # 114: (134.0, 120.0, 54.0),
    # 117: (49.0, 165.0, 42.0),
    # 119: (51.0, 128.0, 133.0),
    # 124: (44.0, 21.0, 163.0),
    # 127: (232.0, 93.0, 193.0),
    # 85: (176.0, 102.0, 54.0),
    # 185: (116.0, 217.0, 17.0),
    # 188: (54.0, 209.0, 150.0),
    # 191: (60.0, 99.0, 204.0),
    # 193: (129.0, 43.0, 144.0),
    # 195: (252.0, 100.0, 106.0),
    # 202: (187.0, 196.0, 73.0),
    # 208: (13.0, 158.0, 40.0),
    # 213: (52.0, 122.0, 152.0),
    # 214: (128.0, 76.0, 202.0),
    # 221: (187.0, 50.0, 115.0),
    # 229: (180.0, 141.0, 71.0),
    # 83: (77.0, 208.0, 35.0),
    # 232: (72.0, 183.0, 168.0),
    # 233: (97.0, 99.0, 203.0),
    # 242: (172.0, 22.0, 158.0),
    # 250: (155.0, 64.0, 40.0),
    # 261: (118.0, 159.0, 30.0),
    # 264: (69.0, 252.0, 148.0),
    # 276: (45.0, 103.0, 173.0),
    # 283: (111.0, 38.0, 149.0),
    # 286: (184.0, 9.0, 49.0),
    # 300: (188.0, 174.0, 67.0),
    # 304: (53.0, 206.0, 53.0),
    # 312: (97.0, 235.0, 252.0),
    # 323: (66.0, 32.0, 182.0),
    # 325: (236.0, 114.0, 195.0),
    # 331: (241.0, 154.0, 83.0),
    # 342: (133.0, 240.0, 52.0),
    # 356: (16.0, 205.0, 144.0),
    # 370: (75.0, 101.0, 198.0),
    # 392: (237.0, 95.0, 251.0),
    # 395: (191.0, 52.0, 49.0),
    # 399: (227.0, 254.0, 54.0),
    # 408: (49.0, 206.0, 87.0),
    # 417: (48.0, 113.0, 150.0),
    # 488: (125.0, 73.0, 182.0),
    # 540: (229.0, 32.0, 114.0),
    # 562: (158.0, 119.0, 28.0),
    # 570: (60.0, 205.0, 27.0),
    # 572: (18.0, 215.0, 201.0),
    # 581: (79.0, 76.0, 153.0),
    # 609: (134.0, 13.0, 116.0),
    # 748: (192.0, 97.0, 63.0),
    # 776: (108.0, 163.0, 18.0),
    # 1156: (95.0, 220.0, 156.0),
    # 1163: (98.0, 141.0, 208.0),
    # 1164: (144.0, 19.0, 193.0),
    # 1165: (166.0, 36.0, 57.0),
    # 1166: (212.0, 202.0, 34.0),
    # 1167: (23.0, 206.0, 34.0),
    # 1168: (91.0, 211.0, 236.0),
    # 81: (79.0, 55.0, 137.0),
    # 1170: (182.0, 19.0, 117.0),
    # 1171: (134.0, 76.0, 14.0),
    # 1172: (87.0, 185.0, 28.0),
    # 1173: (82.0, 224.0, 187.0),
    # 1174: (92.0, 110.0, 214.0),
    # 1175: (168.0, 80.0, 171.0),
    # 1176: (197.0, 63.0, 51.0),
    # 1178: (175.0, 199.0, 77.0),
    # 1179: (62.0, 180.0, 98.0),
    # 1180: (8.0, 91.0, 150.0),
    # 1181: (77.0, 15.0, 130.0),
    # 1182: (154.0, 65.0, 96.0),
    # 1183: (197.0, 152.0, 11.0),
    # 1184: (59.0, 155.0, 45.0),
    # 1185: (12.0, 147.0, 145.0),
    # 1186: (54.0, 35.0, 219.0),
    # 1187: (210.0, 73.0, 181.0),
    # 1188: (221.0, 124.0, 77.0),
    # 1189: (149.0, 214.0, 66.0),
    # 1190: (72.0, 185.0, 134.0),
    # -1: (42.0, 94.0, 198.0)
# }

def map2color(labels):
    output_colors = list()
    # Shuffle the values
    keys = list(SCANNET_COLOR_MAP_200.keys())
    values = list(SCANNET_COLOR_MAP_200.values())
    #random.shuffle(values)   #uncomment to shuffle colors

    # Create a new dictionary with shuffled values
    shuffled_SCANNET_COLOR_MAP_200 = dict(zip(keys, values))
    # for i in [0,1,3,4,5,6,7,8,9]:
    #      shuffled_SCANNET_COLOR_MAP_200[i] = (1, 1, 1)
    
    for label in labels:
        output_colors.append(shuffled_SCANNET_COLOR_MAP_200[label])

    return torch.tensor(output_colors)
@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )

def get_mask_and_scores(
         mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = 'cuda:0'
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if True:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                300, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap
def get_full_res_mask(
         mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        return mask

def load_model_and_data(cfg: DictConfig):
    """
    Simplified function to load a model from a checkpoint and initialize data.

    Args:
        cfg (DictConfig): Configuration object containing model and data paths.

    Returns:
        model: Loaded model instance.
        trainer: PyTorch Lightning Trainer instance.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    seed_everything(cfg.general.seed)

    # Ensure save directory exists
    os.makedirs(cfg.general.save_dir, exist_ok=True)

    # Load model
    logger.info("Initializing model...")
    model = InstanceSegmentation(cfg)
    
    # Load checkpoint if available
    if cfg.general.checkpoint:
        logger.info(f"Loading checkpoint from {cfg.general.checkpoint}...")
        _, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    # Initialize trainer
    trainer = Trainer(
        gpus=cfg.general.gpus,
        default_root_dir=cfg.general.save_dir,
        **cfg.trainer,
    )

    return model, trainer
def load_data_from_npy(path, path2):
    data = np.load(path)
    data2 = np.load(path2)
    return data, data2
class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        full_res_coords=None,
        target_full=None,
        original_colors=None,
        original_normals=None,
        original_coordinates=None,
        idx=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx
def data_preparation(batch):
    (
        coordinates,
        features,
        labels,
        original_labels,
        inverse_maps,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
    ) = ([], [], [], [], [], [], [], [], [])
    voxelization_dict = {
        "ignore_label": 255,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []

    for sample in batch:
        idx.append(sample[7])
        original_coordinates.append(sample[6])
        original_labels.append(sample[2])
        full_res_coords.append(sample[0])
        original_colors.append(sample[4])
        original_normals.append(sample[5])

        coords = np.floor(sample[0] / 0.02)
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample[1],
            }
        )

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            **voxelization_dict
        )
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long())

    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        input_dict["labels"] = labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])

   
    for i in range(len(input_dict["labels"])):
        _, ret_index, ret_inv = np.unique(
            input_dict["labels"][i][:, 0],
            return_index=True,
            return_inverse=True,
        )
        input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
        # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])


    if "labels" in input_dict:
        list_labels = input_dict["labels"]

        target = []
        target_full = []

        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append(
                    {
                        "labels": label_ids,
                        "masks": list_labels[batch_id]
                        == label_ids.unsqueeze(1),
                    }
                )
        else:
            for i in range(len(input_dict["labels"])):
                target.append(
                    {"point2segment": input_dict["labels"][i][:, 0]}
                )
                target_full.append(
                    {
                        "point2segment": torch.from_numpy(
                            original_labels[i][:, 0]
                        ).long()
                    }
                )
            
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    return (
        NoGpu(
            coordinates,
            features,
            original_labels,
            inverse_maps,
            full_res_coords,
            target_full,
            original_colors,
            original_normals,
            original_coordinates,
            idx,
        ),
        target,
        [sample[3] for sample in batch],
    )

def eval_step(batch, model, batch_idx=0):
        data, target, file_names = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        original_coordinates = data.original_coordinates

        # if len(target) == 0 or len(target_full) == 0:
        #    print("no targets")
        #    return None

        if len(data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if True:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device='cuda:0',
        )

        try:
            output = model.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

    

        if True:
            backbone_features = (
                output["backbone_features"].F.detach().cpu().numpy()
            )
            from sklearn import decomposition

            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            rescaled_pca = (
                255
                * (pca_features - pca_features.min())
                / (pca_features.max() - pca_features.min())
            )
        
        eval_instance_step2(
            output,
            target,
            target_full,
            inverse_maps,
            file_names,
            original_coordinates,
            original_colors,
            original_normals,
            raw_coordinates,
            data_idx,
            backbone_features=rescaled_pca,
        )

        
        return 0.0 
def remap_model_output(output):
        label_info = {0: {'color': [139.0, 18.0, 53.0], 'name': 'exclude2', 'validation': True}, 1: {'color': [134.0, 120.0, 54.0], 'name': 'hook_turn', 'validation': True}, 2: {'color': [49.0, 165.0, 42.0], 'name': 'exclude', 'validation': True}, 3: {'color': [51.0, 128.0, 133.0], 'name': 'hook_pull', 'validation': True}, 4: {'color': [44.0, 21.0, 163.0], 'name': 'key_press', 'validation': True}, 5: {'color': [232.0, 93.0, 193.0], 'name': 'rotate', 'validation': True}, 6: {'color': [176.0, 102.0, 54.0], 'name': 'foot_push', 'validation': True}, 7: {'color': [116.0, 217.0, 17.0], 'name': 'unplug', 'validation': True}, 8: {'color': [54.0, 209.0, 150.0], 'name': 'plug_in', 'validation': True}, 9: {'color': [60.0, 99.0, 204.0], 'name': 'pinch_pull', 'validation': True}, 10: {'color': [139.0, 18.0, 53.0], 'name': 'tip_push', 'validation': True}}
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped
def save_visualizations2(
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        sorted_heatmaps=None,
        query_pos=None,
        backbone_features=None,
    ):
        print(sorted_masks)
        print(len(sorted_masks[0]))
        full_res_coords -= full_res_coords.mean(axis=0)

        gt_pcd_pos = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []

        if "labels" in target_full:
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        target_full["labels"].shape[0]
                    )
                )
            )
            for instance_counter, (label, mask) in enumerate(
                zip(target_full["labels"], target_full["masks"])
            ):
                if label == 255:
                    continue

                mask_tmp = mask.detach().cpu().numpy()
                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                if len(mask_coords) == 0:
                    continue

                gt_pcd_pos.append(mask_coords)
                mask_coords_min = full_res_coords[
                    mask_tmp.astype(bool), :
                ].min(axis=0)
                mask_coords_max = full_res_coords[
                    mask_tmp.astype(bool), :
                ].max(axis=0)
                size = mask_coords_max - mask_coords_min
                mask_coords_middle = mask_coords_min + size / 2

                gt_boxes.append(
                    {
                        "position": mask_coords_middle,
                        "size": size,
                        "color": map2color([label])[0],
                    }
                )

                gt_pcd_color.append(
                    map2color([label]).repeat(
                        gt_pcd_pos[-1].shape[0], 1
                    )
                )
                gt_inst_pcd_color.append(
                    instances_colors[instance_counter % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(gt_pcd_pos[-1].shape[0], 1)
                )

                gt_pcd_normals.append(
                    original_normals[mask_tmp.astype(bool), :]
                )

            gt_pcd_pos = np.concatenate(gt_pcd_pos)
            gt_pcd_normals = np.concatenate(gt_pcd_normals)
            gt_pcd_color = np.concatenate(gt_pcd_color)
            gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

        v = vis.Visualizer()

        v.add_points(
            "RGB Input",
            full_res_coords,
            colors=original_colors,
            normals=original_normals,
            visible=True,
            point_size=point_size,
        )

        if backbone_features is not None:
            v.add_points(
                "PCA",
                full_res_coords,
                colors=backbone_features,
                normals=original_normals,
                visible=False,
                point_size=point_size,
            )

        if "labels" in target_full:
            v.add_points(
                "Semantics (GT)",
                gt_pcd_pos,
                colors=gt_pcd_color,
                normals=gt_pcd_normals,
                alpha=0.8,
                visible=False,
                point_size=point_size,
            )
            v.add_points(
                "Instances (GT)",
                gt_pcd_pos,
                colors=gt_inst_pcd_color,
                normals=gt_pcd_normals,
                alpha=0.8,
                visible=False,
                point_size=point_size,
            )

        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []

        for did in range(len(sorted_masks)):
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        max(1, sorted_masks[did].shape[1])
                    )
                )
            )

            for i in reversed(range(sorted_masks[did].shape[1])):
                coords = full_res_coords[
                    sorted_masks[did][:, i].astype(bool), :
                ]

                mask_coords = full_res_coords[
                    sorted_masks[did][:, i].astype(bool), :
                ]
                mask_normals = original_normals[
                    sorted_masks[did][:, i].astype(bool), :
                ]

                label = sort_classes[did][i]

                if len(mask_coords) == 0:
                    continue

                pred_coords.append(mask_coords)
                pred_normals.append(mask_normals)

                pred_sem_color.append(
                    map2color([label]).repeat(
                        mask_coords.shape[0], 1
                    )
                )

                pred_inst_color.append(
                    instances_colors[i % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(mask_coords.shape[0], 1)
                )

            if len(pred_coords) > 0:
                pred_coords = np.concatenate(pred_coords)
                pred_normals = np.concatenate(pred_normals)
                pred_sem_color = np.concatenate(pred_sem_color)
                pred_inst_color = np.concatenate(pred_inst_color)
                unique_values, counts = np.unique(pred_sem_color, axis=0, return_counts=True)

                # Convert rows to tuples to use as dictionary keys
                unique_counts_dict = {tuple(row): count for row, count in zip(unique_values, counts)}
                print("Unique values and counts:")
                pprint(unique_counts_dict)
                v.add_points(
                    "Semantics (Mask3D)",
                    pred_coords,
                    colors=pred_sem_color,
                    normals=None,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )
                v.add_points(
                    "Instances (Mask3D)",
                    pred_coords,
                    colors=pred_inst_color,
                    normals=None,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )

        v.save(
            f"/home/gokul/test_eval/visualizations/{file_name}"
        )

def eval_instance_step2(
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = 0
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[-1][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[-1]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[-1]["pred_masks"])):
            if not first_full_res:
                
                masks = (
                    prediction[-1]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )

                
                new_preds = {
                    "pred_masks": list(),
                    "pred_logits": list(),
                }

                curr_coords_idx = masks.shape[0]
                curr_coords = raw_coords[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx
                ]
                offset_coords_idx += curr_coords_idx

                for curr_query in range(masks.shape[1]):
                    curr_masks = masks[:, curr_query] > 0

                    if curr_coords[curr_masks].shape[0] > 0:
                        clusters = (
                            DBSCAN(
                                eps=0.95,
                                min_samples=1,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[curr_masks])
                            .labels_
                        )

                        new_mask = torch.zeros(curr_masks.shape, dtype=int)
                        new_mask[curr_masks] = (
                            torch.from_numpy(clusters) + 1
                        )

                        for cluster_id in np.unique(clusters):
                            original_pred_masks = masks[:, curr_query]
                            if cluster_id != -1:
                                new_preds["pred_masks"].append(
                                    original_pred_masks
                                    * (new_mask == cluster_id + 1)
                                )
                                new_preds["pred_logits"].append(
                                    prediction[-1][
                                        "pred_logits"
                                    ][bid, curr_query]
                                )

                scores, masks, classes, heatmap = get_mask_and_scores(
                    torch.stack(new_preds["pred_logits"]).cpu(),
                    torch.stack(new_preds["pred_masks"]).T,
                    len(new_preds["pred_logits"]),
                    11-1,
                )
                

                masks = get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"
                masks = self.get_full_res_mask(
                    prediction[self.decoder_id]["pred_masks"][bid].cpu(),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]["pred_logits"][bid].cpu(),
                    masks,
                    prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                    self.model.num_classes - 1,
                    device="cpu",
                )

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            
            all_pred_classes.append(sort_classes)
            all_pred_masks.append(sorted_masks)
            all_pred_scores.append(sort_scores_values)
            all_heatmaps.append(sorted_heatmap)

        # if self.validation_dataset.dataset_name == "scannet200":
        #     all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
        #     if self.config.data.test_mode != "test":
        #         target_full_res[bid]["labels"][
        #             target_full_res[bid]["labels"] == 0
        #         ] = -1

        for bid in range(len(prediction[-1]["pred_masks"])):
            all_pred_classes[
                bid
            ] = remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            

            preds = {}
                # prev val_dataset
            preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }

            
                
               
            save_visualizations2(
                target_full_res[bid],
                full_res_coords[bid],
                [preds[file_names[bid]]["pred_masks"]],
                [preds[file_names[bid]]["pred_classes"]],
                file_names[bid],
                original_colors[bid],
                original_normals[bid],
                [preds[file_names[bid]]["pred_scores"]],
                sorted_heatmaps=[all_heatmaps[bid]],
                query_pos=all_query_pos[bid]
                if len(all_query_pos) > 0
                else None,
                backbone_features=backbone_features,
                point_size=20,
            )

            # if self.config.general.export:
            #     if self.validation_dataset.dataset_name == "stpls3d":
            #         scan_id, _, _, crop_id = file_names[bid].split("_")
            #         crop_id = int(crop_id.replace(".txt", ""))
            #         file_name = (
            #             f"{scan_id}_points_GTv3_0{crop_id}_inst_nostuff"
            #         )

            #         self.export(
            #             self.preds[file_names[bid]]["pred_masks"],
            #             self.preds[file_names[bid]]["pred_scores"],
            #             self.preds[file_names[bid]]["pred_classes"],
            #             file_name,
            #             self.decoder_id,
            #         )
            #     else:
            #         self.export(
            #             self.preds[file_names[bid]]["pred_masks"],
            #             self.preds[file_names[bid]]["pred_scores"],
            #             self.preds[file_names[bid]]["pred_classes"],
            #             file_names[bid],
            #             self.decoder_id,
            #         )

# @hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig, cordinates_to_eval, colors_to_eval, normals_to_eval):
    """
    Main function for loading model and data using the configuration loaded from YAML.

    Args:
        cfg (DictConfig): Configuration object containing model and training settings.
    """
    # Load model and trainer
    model, trainer = load_model_and_data(cfg)
    checkpoint_path = "saved/SIR_dummyClass/last-epoch.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load the model state
    model.load_state_dict(checkpoint['state_dict'])  # Adjust the key if needed
    model = model.to('cuda:0')
 
    labels = np.zeros((cordinates_to_eval.shape[0],3))
    colors = np.repeat([[-1.8586, -1.6315, -1.4888]], cordinates_to_eval.shape[0], axis=0)
    batch = [(cordinates_to_eval,np.concatenate((colors, normals_to_eval),axis=1), labels,'test', colors,normals_to_eval,cordinates_to_eval,1 )]
    with torch.no_grad():
        dataaaa = data_preparation(batch)
        model2 = hydra.utils.instantiate(cfg.model)
        eval_step(dataaaa, model)
    exit()
    voxelization_dict = {
        "ignore_label": 255,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }
    
    with torch.no_grad():
        # Depending on your model's input format, process the data accordingly
        # input_data should be a batch or preprocessed image/tensor
        data_cordinates = torch.tensor(cordinates_to_eval)
        values = torch.tensor([-1.8586, -1.6315, -1.4888])
        feats = np.concatenate((colors_to_eval, normals_to_eval, cordinates_to_eval), axis=1)

        data_features = torch.tensor(feats)
        
        data_cordinates = np.floor(cordinates_to_eval/ 0.02)

        
        # old_cords = coordinates
        # data = ME.SparseTensor(
        #     coordinates=coordinates,
        #     features=features,
        #     device='cuda:0'        )
        
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(data_cordinates).to("cpu").contiguous(),
                "features": colors_to_eval,
            }
        )

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            **voxelization_dict
        )
        
        featuresss = values.repeat(cordinates_to_eval.shape[0], 1)
        input_dict = {"coords": [torch.from_numpy(data_cordinates[unique_map])], "feats": [data_features[unique_map]]}
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        raw_coordinates = features[:, -3:]
        data = ME.SparseTensor(
            coordinates=coordinates,
            features=featuresss[unique_map],
            device='cuda:0'        )

        predictions = model(data, None, raw_coordinates=raw_coordinates.float(),
                is_eval=True)  # Modify based on your model's forward method
        
        run_vis(predictions, inverse_map, cordinates_to_eval, colors_to_eval, normals_to_eval)
    return predictions

    
    # Here, you can load your dataset and start training or testing
    # trainer.fit(model, datamodule=your_data_module)  # Example of fitting the model
    # or
    # trainer.test(model, datamodule=your_data_module)  # Example of testing the model

