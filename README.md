# Lite_Rebar_Detector
![img1](result/1587035705.2716005.png)

使用自定议小网络，在 [智能盘点—钢筋数量AI识别](https://www.datafountain.cn/competitions/332/details)
使用 [Objects as Points](https://github.com/zengliangjun/CenterNet.git)，自定义小型网络。
因为没有评估数据集， 所以训练时使用原训练数据集进行扩张数据进行训练，评估时使用原训练集进行评估。

评估效果达到 {'ap': array([0.99799332]), 'map': 0.9979933217444574}

- 模型文件大小： 1.8 MB (1,822,205 bytes)

- Total params: 361,879
- Total memory: 576.92MB
- Total MAdd: 9.79GMAdd
- Total Flops: 4.91GFlops
- Total MemR+W: 1.15GB

## Tested the environment that works
- Ubuntu16.04
- Python3.7
- onnxruntime1.2
- Support for onnx inference and enval.

## 

- 获取代码

```
git clone https://github.com/zengliangjun/Lite_Rebar_Detector.git
cd Lite_Rebar_Detector/src/
```

- 测试

```Python
python demo.py --task ctdetv2 --dataset rebar --resume --gpus 0 --debug 2 \
               --demo '../images' \
               --load_model '../exp/ctdetv2/rebaraug_lite/model_best.onnx'
```
- 评估

先修改 src/lib/datasets/dataset/rebar.py 中
_label_file
_image_root 在地的路径

```Python
python eval.py --task ctdetv2 --dataset rebar --resume --gpus 0 \
               --load_model '../exp/ctdetv2/rebaraug_lite/model_best.onnx'
```

##  Reference
- [Objects as Points](https://github.com/zengliangjun/CenterNet.git)

![img1](result/1587035707.9053595.png)