from os import path as osp
from pathlib import Path

import mmengine

total_num = {
    0: 4541,
    1: 1101,
    2: 4661,
    3: 801,
    4: 271,
    5: 2761,
    6: 1101,
    7: 1101,
    8: 4071,
    9: 1591,
    10: 1201,
    11: 921,
    12: 1061,
    13: 3281,
    14: 631,
    15: 1901,
    16: 1731,
    17: 491,
    18: 1801,
    19: 4981,
    20: 831,
    21: 2721,
# DriveOSS - ID 
    311: 101,
    322: 300,
    312: 184,
    313: 400,
    314: 399,
    315: 212,
    316: 412,
    317: 79,
    318: 186,
    319: 165,
    320: 253,
    321: 236,
# CUDAL
    30: 2500,
    31: 2500,
    36: 2500,
    40: 2500,
    41: 2188,
# DriveOSS - OOD 
    411: 370,
    412: 377,
    413: 328,
    414: 351,
    421: 151,
    422: 136,
    423: 181,
    424: 176,
    431: 414,
    432: 328,
    433: 312,
    434: 354,
    441: 365,
    442: 331,
    443: 256,
    444: 362,
    451: 126,
    452: 112,
    453: 191,
    454: 161,
}

fold_split = {
    # 'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 311, 312, 313, 314, 315, 321, 322, 30, 31, 36, 40, 41],
    'val': [316, 317, 318, 319, 320],
    #'val': [413, 414, 423, 424, 433, 434, 443, 444, 453, 454],
    # 'val': [8],
    #'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    #'trainval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 311, 312, 313, 314, 315, 321,316, 317, 318, 319, 320, 322]
    # 'trainval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}


def get_semantickitti_info(split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'SemanticKITTI'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'sequences/00/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'pts_panoptic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'sample_id': '00'
                },
                ...
            }
        }
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='SemanticKITTI')
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(0, total_num[i_folder]):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'velodyne',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'pts_panoptic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_id':
                str(i_folder) + str(j)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(pkl_prefix, save_path):
    """Create info file of SemanticKITTI dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info.')
    save_path = Path(save_path)

    semantickitti_infos_train = get_semantickitti_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SemanticKITTI info train file is saved to {filename}')
    mmengine.dump(semantickitti_infos_train, filename)
    semantickitti_infos_val = get_semantickitti_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SemanticKITTI info val file is saved to {filename}')
    mmengine.dump(semantickitti_infos_val, filename)
    semantickitti_infos_test = get_semantickitti_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SemanticKITTI info test file is saved to {filename}')
    mmengine.dump(semantickitti_infos_test, filename)
    semantickitti_infos_test = get_semantickitti_info(split='trainval')
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'SemanticKITTI info test file is saved to {filename}')
    mmengine.dump(semantickitti_infos_test, filename)