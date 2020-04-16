from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
try:
  from .sample.ctdetv2 import CTDetDatasetV2
  from .dataset.rebar import Rebar, RebarAug
except:
  from datasets.sample.ctdetv2 import CTDetDatasetV2
  from datasets.dataset.rebar import Rebar, RebarAug


dataset_factory = {
  'rebar': Rebar,
  'rebaraug': RebarAug
}

_sample_factory = {
  'ctdetv2': CTDetDatasetV2,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
from opts import opts

if __name__ == '__main__':
    if False:
        opt = opts().parse()
        Dataset = get_dataset('rebar', 'ctdetv2')
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        print(opt)

        db = Dataset(opt, 'val')
        _classes = db.class_name
        print(len(_classes))

        for _id in range(len(db)):
          _item = db[_id]
          print(_item)

    if True:
        opt = opts().parse()

        #Dataset = get_dataset('coco', 'ctv2det')
        Dataset = get_dataset('rebar', 'ctdetv2')
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        print(opt)

        db = Dataset(opt, 'train')
        _classes = db.class_name
        print(_classes, len(_classes))

        for _id in range(len(db)):
          _item = db[_id]

          inp = _item['input']

          inp = inp.transpose(1, 2, 0)
          inp = (inp * db.std + db.mean)

          import matplotlib.pyplot as plt
          plt.figure(figsize=(6.4, 4.8))
          plt.subplot(2, 2, 1)
          plt.title('image')
          plt.imshow(inp)

          if 'neg_pos_hm' in _item:
            neg_pos_hm = _item['neg_pos_hm']
            plt.subplot(2, 2, 3)
            plt.title('neg')
            plt.imshow(neg_pos_hm[0])
            plt.subplot(2, 2, 4)
            plt.title('pos')
            plt.imshow(neg_pos_hm[1])
          else:
            hm = _item['hm']
            plt.subplot(2, 2, 3)
            plt.title('neg')
            plt.imshow(hm[0])
            plt.subplot(2, 2, 4)
            plt.title('pos')
            plt.imshow(hm[1])
          plt.show()
