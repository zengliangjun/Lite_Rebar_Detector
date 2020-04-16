import os.path as osp
import sys
import csv
import numpy as np
import cv2
import glob
from abc import abstractmethod

INVALID_IMG = ['E42F504E.jpg']

class RebarBase():

    num_classes = 2
    default_resolution = [1024, 1152]#[512, 576]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)
    max_objs = 1024

    def __init__(self, opt, split = 'test'):
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.split = split
        self.opt = opt

        self.class_name = ['__background__', 'rebar']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                    for v in range(1, self.num_classes + 1)]

        self.max_per_image = RebarBase.max_objs
        self._init()

    @abstractmethod
    def _init(self):
        pass

    def __len__(self):
        return len(self.labels)

    def getitem(self, idx):
        _file_name, _bboxs = self.labels[idx]
        _path = osp.join(self.image_root, _file_name)
        _img = cv2.imread(_path)

        return _img, _bboxs, 1

    @staticmethod
    def _open_for_csv(path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


class Rebar(RebarBase):
    def __init__(self, opt, split = 'test'):
        super(Rebar, self).__init__(opt, split)

    def _init(self):
        _label_file = '/workspace/data0/rebar/train_labels.csv'
        _image_root = '/workspace/data0/rebar/train'

        self.image_root = _image_root

        _labels = []
        with Rebar._open_for_csv(_label_file) as _file:
            csv_reader = csv.reader(_file, delimiter=',')

            _file_name = None
            _bboxs = []

            for line, row in enumerate(csv_reader):
                if line == 0:
                    continue

                _name = row[0].strip()
                _bbox = row[1].split()
                _bbox = np.array([float(x) for x in _bbox], dtype=float)

                if _file_name is None:
                    _file_name = _name
                    _bboxs.append(_bbox)

                elif _name == _file_name:

                    _bboxs.append(_bbox)
                else:
                    if _file_name not in INVALID_IMG:
                        _labels.append([_file_name, _bboxs])

                    _file_name = _name
                    _bboxs = []
                    _bboxs.append(_bbox)

        self.labels = _labels
        import random
        random.shuffle(self.labels)

class RebarAug(RebarBase):
    def __init__(self, opt, split = 'test'):
        super(RebarAug, self).__init__(opt, split)

    def _init(self):
        if self.split == 'val':
            self._init_org()
        else:
            self._init_aug()

    def _init_org(self):
        _label_file = '/workspace/data0/rebar/train_labels.csv'
        _image_root = '/workspace/data0/rebar/train'

        self.image_root = _image_root

        _labels = []
        with Rebar._open_for_csv(_label_file) as _file:
            csv_reader = csv.reader(_file, delimiter=',')

            _file_name = None
            _bboxs = []

            for line, row in enumerate(csv_reader):
                if line == 0:
                    continue

                _name = row[0].strip()
                _bbox = row[1].split()
                _bbox = np.array([float(x) for x in _bbox], dtype=float)

                if _file_name is None:
                    _file_name = _name
                    _bboxs.append(_bbox)

                elif _name == _file_name:

                    _bboxs.append(_bbox)
                else:
                    if _file_name not in INVALID_IMG:
                        _labels.append([_file_name, _bboxs])

                    _file_name = _name
                    _bboxs = []
                    _bboxs.append(_bbox)

        self.labels = _labels
        import random
        random.shuffle(self.labels)


    def _init_aug(self):
        _image_root = '/workspace/data0/rebar/aug'
        self.image_root = _image_root

        _files = glob.glob(osp.join(_image_root, '*.jpg'))
        _labels = []
        for _file in _files:
            _file_name = osp.basename(_file)
            _label_file = _file.replace('jpg', 'txt')

            _bboxs = []
            with open(_label_file) as _fd:
                _lines = _fd.readlines()
                for line in _lines:

                    _bbox = line.split()[1:]
                    _bbox = np.array([float(x) for x in _bbox], dtype=float)

                    _bboxs.append(_bbox)

            _labels.append([_file_name, _bboxs])

        self.labels = _labels
        import random
        random.shuffle(self.labels)

if __name__ == '__main__':
    rebar = RebarAug(None)
    _widths = {}
    _heights = {}

    _scale = []
    for _id in range(len(rebar)):
        _img, _bboxs, _ = rebar.getitem(_id)

        for _box in _bboxs:
            cv2.rectangle(_img, (int(_box[0]), int(_box[1])), (int(_box[2]), int(_box[3])), (0, 0, 255), thickness=4)

        _img = cv2.resize(_img, (_img.shape[1] // 5 * 2, _img.shape[0] // 5 * 2))
        cv2.imshow('img', _img)

        key = cv2.waitKey(0)
        if key == 27:
            exit(-1)

        if key == 99: #key c
            continue
