### This file contains the HEAD - COMMON - TAIL split category ids for ScanNet 200

HEAD_CATS_SCANNET_200 = [

'hook_turn',
'exclude',
'hook_pull',
'rotate',
'foot_push',
'pinch_pull'
]
COMMON_CATS_SCANNET_200 = [
   'tip_push',
   'key_press'
]
TAIL_CATS_SCANNET_200 = [
'unplug',
'plug_in'
]


### Given the different size of the official train and val sets, not all ScanNet200 categories are present in the validation set.
### Here we list of categories with labels and IDs present in both train and validation set, and the remaining categories those are present in train, but not in val
### We dont evaluate on unseen validation categories in this benchmark

VALID_CLASS_IDS_200_VALIDATION = (
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
)

CLASS_LABELS_200_VALIDATION = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
)

VALID_CLASS_IDS_200_TRAIN_ONLY = (
    "bicycle",
    "storage container",
    "candle",
    "guitar case",
    "purse",
    "alarm clock",
    "music stand",
    "cd case",
    "structure",
    "storage organizer",
    "luggage",
)

CLASS_LABELS_200_TRAIN_ONLY = (
    121,
    221,
    286,
    331,
    399,
    572,
    581,
    1174,
    1178,
    1183,
    1190,
)
