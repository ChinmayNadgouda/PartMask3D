import os
import pyviz3d.visualizer as vis

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
from trainer.trainer import InstanceSegmentation
from utils.utils import load_checkpoint_with_missing_or_exsessive_keys

SCANNET_COLOR_MAP_200 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (188.0, 189.0, 34.0),
    3: (152.0, 223.0, 138.0),
    4: (255.0, 152.0, 150.0),
    5: (214.0, 39.0, 40.0),
    6: (91.0, 135.0, 229.0),
    7: (31.0, 119.0, 180.0),
    8: (229.0, 91.0, 104.0),
    9: (247.0, 182.0, 210.0),
    10: (91.0, 229.0, 110.0),
    11: (255.0, 187.0, 120.0),
    13: (141.0, 91.0, 229.0),
    14: (112.0, 128.0, 144.0),
    15: (196.0, 156.0, 148.0),
    16: (197.0, 176.0, 213.0),
    17: (44.0, 160.0, 44.0),
    18: (148.0, 103.0, 189.0),
    19: (229.0, 91.0, 223.0),
    21: (219.0, 219.0, 141.0),
    22: (192.0, 229.0, 91.0),
    23: (88.0, 218.0, 137.0),
    24: (58.0, 98.0, 137.0),
    26: (177.0, 82.0, 239.0),
    27: (255.0, 127.0, 14.0),
    28: (237.0, 204.0, 37.0),
    29: (41.0, 206.0, 32.0),
    31: (62.0, 143.0, 148.0),
    32: (34.0, 14.0, 130.0),
    33: (143.0, 45.0, 115.0),
    34: (137.0, 63.0, 14.0),
    35: (23.0, 190.0, 207.0),
    36: (16.0, 212.0, 139.0),
    38: (90.0, 119.0, 201.0),
    39: (125.0, 30.0, 141.0),
    40: (150.0, 53.0, 56.0),
    41: (186.0, 197.0, 62.0),
    42: (227.0, 119.0, 194.0),
    44: (38.0, 100.0, 128.0),
    45: (120.0, 31.0, 243.0),
    46: (154.0, 59.0, 103.0),
    47: (169.0, 137.0, 78.0),
    48: (143.0, 245.0, 111.0),
    49: (37.0, 230.0, 205.0),
    50: (14.0, 16.0, 155.0),
    51: (196.0, 51.0, 182.0),
    52: (237.0, 80.0, 38.0),
    54: (138.0, 175.0, 62.0),
    55: (158.0, 218.0, 229.0),
    56: (38.0, 96.0, 167.0),
    57: (190.0, 77.0, 246.0),
    58: (208.0, 49.0, 84.0),
    59: (208.0, 193.0, 72.0),
    62: (55.0, 220.0, 57.0),
    63: (10.0, 125.0, 140.0),
    64: (76.0, 38.0, 202.0),
    65: (191.0, 28.0, 135.0),
    66: (211.0, 120.0, 42.0),
    67: (118.0, 174.0, 76.0),
    68: (17.0, 242.0, 171.0),
    69: (20.0, 65.0, 247.0),
    70: (208.0, 61.0, 222.0),
    71: (162.0, 62.0, 60.0),
    72: (210.0, 235.0, 62.0),
    73: (45.0, 152.0, 72.0),
    74: (35.0, 107.0, 149.0),
    75: (160.0, 89.0, 237.0),
    76: (227.0, 56.0, 125.0),
    77: (169.0, 143.0, 81.0),
    78: (42.0, 143.0, 20.0),
    79: (25.0, 160.0, 151.0),
    80: (82.0, 75.0, 227.0),
    82: (253.0, 59.0, 222.0),
    84: (240.0, 130.0, 89.0),
    86: (123.0, 172.0, 47.0),
    87: (71.0, 194.0, 133.0),
    88: (24.0, 94.0, 205.0),
    89: (134.0, 16.0, 179.0),
    90: (159.0, 32.0, 52.0),
    93: (213.0, 208.0, 88.0),
    95: (64.0, 158.0, 70.0),
    96: (18.0, 163.0, 194.0),
    97: (65.0, 29.0, 153.0),
    98: (177.0, 10.0, 109.0),
    99: (152.0, 83.0, 7.0),
    100: (83.0, 175.0, 30.0),
    101: (18.0, 199.0, 153.0),
    102: (61.0, 81.0, 208.0),
    103: (213.0, 85.0, 216.0),
    104: (170.0, 53.0, 42.0),
    105: (161.0, 192.0, 38.0),
    106: (23.0, 241.0, 91.0),
    107: (12.0, 103.0, 170.0),
    110: (151.0, 41.0, 245.0),
    112: (133.0, 51.0, 80.0),
    115: (184.0, 162.0, 91.0),
    116: (50.0, 138.0, 38.0),
    118: (31.0, 237.0, 236.0),
    120: (0, 0, 0),
    121: (0, 0, 0),
    122: (0, 0, 0),
    125: (0, 0, 0),
    128: (0, 0, 0),
    126: (0, 0, 0),
    123: (0, 0, 0),
    129: (0, 0, 0),
    12: (250.0, 253.0, 26.0),
    20: (55.0, 191.0, 73.0),
    25: (60.0, 126.0, 146.0),
    30: (153.0, 108.0, 234.0),
    37: (184.0, 58.0, 125.0),
    43: (135.0, 84.0, 14.0),
    53: (139.0, 248.0, 91.0),
    60: (53.0, 200.0, 172.0),
    61: (63.0, 69.0, 134.0),
    91: (190.0, 75.0, 186.0),
    92: (127.0, 63.0, 52.0),
    94: (141.0, 182.0, 25.0),
    108: (56.0, 144.0, 89.0),
    109: (64.0, 160.0, 250.0),
    111: (182.0, 86.0, 245.0),
    113: (139.0, 18.0, 53.0),
    114: (134.0, 120.0, 54.0),
    117: (49.0, 165.0, 42.0),
    119: (51.0, 128.0, 133.0),
    124: (44.0, 21.0, 163.0),
    127: (0, 0, 0),
    85: (176.0, 102.0, 54.0),
    185: (116.0, 217.0, 17.0),
    188: (54.0, 209.0, 150.0),
    191: (60.0, 99.0, 204.0),
    193: (129.0, 43.0, 144.0),
    195: (252.0, 100.0, 106.0),
    202: (187.0, 196.0, 73.0),
    208: (13.0, 158.0, 40.0),
    213: (52.0, 122.0, 152.0),
    214: (128.0, 76.0, 202.0),
    221: (187.0, 50.0, 115.0),
    229: (180.0, 141.0, 71.0),
    230: (77.0, 208.0, 35.0),
    232: (72.0, 183.0, 168.0),
    233: (97.0, 99.0, 203.0),
    242: (172.0, 22.0, 158.0),
    250: (155.0, 64.0, 40.0),
    261: (118.0, 159.0, 30.0),
    264: (69.0, 252.0, 148.0),
    276: (45.0, 103.0, 173.0),
    283: (111.0, 38.0, 149.0),
    286: (184.0, 9.0, 49.0),
    300: (188.0, 174.0, 67.0),
    304: (53.0, 206.0, 53.0),
    81: (97.0, 235.0, 252.0),
    323: (66.0, 32.0, 182.0),
    325: (236.0, 114.0, 195.0),
    331: (241.0, 154.0, 83.0),
    342: (133.0, 240.0, 52.0),
    356: (16.0, 205.0, 144.0),
    370: (75.0, 101.0, 198.0),
    392: (237.0, 95.0, 251.0),
    395: (191.0, 52.0, 49.0),
    399: (227.0, 254.0, 54.0),
    408: (49.0, 206.0, 87.0),
    417: (48.0, 113.0, 150.0),
    488: (125.0, 73.0, 182.0),
    540: (229.0, 32.0, 114.0),
    562: (158.0, 119.0, 28.0),
    570: (60.0, 205.0, 27.0),
    572: (18.0, 215.0, 201.0),
    581: (79.0, 76.0, 153.0),
    609: (134.0, 13.0, 116.0),
    748: (192.0, 97.0, 63.0),
    776: (108.0, 163.0, 18.0),
    1156: (95.0, 220.0, 156.0),
    1163: (98.0, 141.0, 208.0),
    1164: (144.0, 19.0, 193.0),
    1165: (166.0, 36.0, 57.0),
    1166: (212.0, 202.0, 34.0),
    1167: (23.0, 206.0, 34.0),
    1168: (91.0, 211.0, 236.0),
    1169: (79.0, 55.0, 137.0),
    1170: (182.0, 19.0, 117.0),
    1171: (134.0, 76.0, 14.0),
    1172: (87.0, 185.0, 28.0),
    1173: (82.0, 224.0, 187.0),
    1174: (92.0, 110.0, 214.0),
    1175: (168.0, 80.0, 171.0),
    1176: (197.0, 63.0, 51.0),
    1178: (175.0, 199.0, 77.0),
    1179: (62.0, 180.0, 98.0),
    1180: (8.0, 91.0, 150.0),
    1181: (77.0, 15.0, 130.0),
    1182: (154.0, 65.0, 96.0),
    1183: (197.0, 152.0, 11.0),
    1184: (59.0, 155.0, 45.0),
    1185: (12.0, 147.0, 145.0),
    1186: (54.0, 35.0, 219.0),
    1187: (210.0, 73.0, 181.0),
    1188: (221.0, 124.0, 77.0),
    1189: (149.0, 214.0, 66.0),
    1190: (72.0, 185.0, 134.0),
    -1: (42.0, 94.0, 198.0),
}

def map2color(labels):
    print('msdfsdfsdfsdf',labels)
    output_colors = list()

    for label in labels:
        output_colors.append(SCANNET_COLOR_MAP_200[label])

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

def run_vis(output, inverse_map, org_cords, org_colors, org_normals):
        
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
        print(type(org_cords))
        eval_instance_step(
            output,
            None,
            None,
            [inverse_map],
            ['test'],
            [org_cords],
            [org_colors],
            [org_normals],
            org_cords,
            1,
            backbone_features=rescaled_pca
        )

def eval_instance_step(
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
        predss = {}
        decoder_id = -1 
        # label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[decoder_id]["pred_masks"])):
            if not first_full_res:

                masks = (
                    prediction[decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )
                print('TESTSTSTSTSTST',masks.shape[1])

                if True:
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
                                    eps=0.85,
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
                                        prediction[decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )

                    scores, masks, classes, heatmap = get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        127 - 1,
                    )
                else:
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        self.model.num_classes - 1,
                    )

                masks = get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    None,
                )

                heatmap = get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    None,
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        None,
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"


            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            print(len(masks))

            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            
            all_pred_classes.append(sort_classes)
            all_pred_masks.append(sorted_masks)
            all_pred_scores.append(sort_scores_values)
            all_heatmaps.append(sorted_heatmap)


            all_pred_classes[bid][all_pred_classes[bid] == 0] = -1

        for bid in range(len(prediction[decoder_id]["pred_masks"])):
            # all_pred_classes[
            #     bid
            # ] = self.validation_dataset._remap_model_output(
            #     all_pred_classes[bid].cpu() + label_offset
            # )

            
           
            predss[file_names[bid]] = {
                "pred_masks": all_pred_masks[bid],
                "pred_scores": all_pred_scores[bid],
                "pred_classes": all_pred_classes[bid],
            }


            save_visualizations(
                None,
                full_res_coords[bid],
                [predss[file_names[bid]]["pred_masks"]],
                [predss[file_names[bid]]["pred_classes"]],
                file_names[bid],
                original_colors[bid],
                original_normals[bid],
                [predss[file_names[bid]]["pred_scores"]],
                sorted_heatmaps=[all_heatmaps[bid]],
                query_pos=all_query_pos[bid]
                if len(all_query_pos) > 0
                else None,
                backbone_features=backbone_features,
                point_size=20,
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
                150, sorted=True
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
        print(inverse_map)
        mask = mask.detach().cpu()[inverse_map]  # full res

        return mask
def save_visualizations(
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

        full_res_coords -= full_res_coords.mean(axis=0)

        gt_pcd_pos = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []


        v = vis.Visualizer()

        v.add_points(
            "RGB Input",
            full_res_coords,
            colors=original_colors,
            normals=original_normals,
            visible=True,
            point_size=point_size,
        )

        # if backbone_features is not None:
        #     v.add_points(
        #         "PCA",
        #         full_res_coords,
        #         colors=backbone_features,
        #         normals=original_normals,
        #         visible=False,
        #         point_size=point_size,
        #     )

        
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
                print(label.cpu().detach().numpy())
                pred_sem_color.append(
                    map2color([int(label.cpu().detach().numpy())]).repeat(
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

                v.add_points(
                    "Semantics (Mask3D)",
                    pred_coords,
                    colors=pred_sem_color,
                    normals=pred_normals,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )
                v.add_points(
                    "Instances (Mask3D)",
                    pred_coords,
                    colors=pred_inst_color,
                    normals=pred_normals,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )

        v.save(
            f"/home/gokul/visualizations/{file_name}"
        )
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
# @hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig, cordinates_to_eval, colors_to_eval, normals_to_eval):
    """
    Main function for loading model and data using the configuration loaded from YAML.

    Args:
        cfg (DictConfig): Configuration object containing model and training settings.
    """
    # Load model and trainer
    model, trainer = load_model_and_data(cfg)
    checkpoint_path = "/home/gokul/ConceptGraphs/concept-graphs/conceptgraph/Mask3D/saved/working_checkpoints/last-epoch.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load the model state
    model.load_state_dict(checkpoint['state_dict'])  # Adjust the key if needed
    model.eval()
    model = model.to('cuda:0')
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

