from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
from opts import opts
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset

from utils.voc_cal_ap import eval_detection_voc

def eval(opt):
    config = opt.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus_str
    Dataset = get_dataset(config.dataset, config.task)
    _db = Dataset(config, 'val')
    config = opt.update_dataset_info_and_set_heads(config, _db)


    if config.task in detector_factory:
        Detector = detector_factory[config.task]
    else:
        for _key in detector_factory.keys():
            if config.task.startswith(_key):
                Detector = detector_factory[_key]
                break

    detector = Detector(config)

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxs_list = []
    gt_labels_list = []
    gt_difficult_list = []

    for _id in range(len(_db)):
        _img, _bboxs, _c = _db.getitem(_id)

        _labels = [_c - 1 for _id in range(len(_bboxs))]
        _difficults = [0 for _id in range(len(_bboxs))]

        gt_boxs_list.append(np.array(_bboxs, dtype=np.float))
        gt_labels_list.append(np.array(_labels))
        gt_difficult_list.append(np.array(_difficults))

        ret = detector.run(_img)


        _pred_boxes = []
        _pred_labels = []
        _pred_scores = []

        for j in range(1, detector.num_classes + 1):
            results = ret["results"][j]
            for bbox in results:
                if bbox[4] > config.vis_thresh:
                    _pred_boxes.append(bbox[:4])
                    _pred_labels.append(j - 1)
                    _pred_scores.append(bbox[4])

        pred_boxes_list.append(np.array(_pred_boxes, dtype=np.float))
        pred_labels_list.append(np.array(_pred_labels))
        pred_scores_list.append(np.array(_pred_scores))

        #break

    result = eval_detection_voc(
        pred_boxes_list,
        pred_labels_list,
        pred_scores_list,
        gt_boxs_list,
        gt_labels_list)

    print(result)

if __name__ == '__main__':
    opt = opts()
    eval(opt)
