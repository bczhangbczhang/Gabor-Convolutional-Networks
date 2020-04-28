# Gabor CNNs, Gabor Convolutional Networks

We provide a demo for GCN on MNIST dataset based on **torch7** and also **PyTorch**.
THere is a pytorch version shown in https://github.com/jxgu1016/Gabor_CNN_PyTorch

To run this demo, you should  install these dependencies:
```
luarocks install torchnet
luarocks install optnet
```

install GCN:
```bash
cd $DIR/GCN
bash install.sh
```

run this demo:
```bash
cd $DIR/MNIST_demo
bash ./scripts/Train_MNIST.sh
```
To run the pytorch branch, please refer to [this branch](https://github.com/bczhangbczhang/Gabor-Convolutional-Networks/tree/pytorch).

The experimental results of the comparison algorithm are directly quoted from the corresponding papers. 

## Acknowledgement
This demo is partially based on the code of [Orientation Response Networks](http://zhouyanzhao.github.io/ORN/).
If you use this demo please cite our paper and ORN. 

## Bibtex

@article{GaborCNNs, title={Gabor Convolutional Networks}, author={Luan, Shangzhen and chen, chen and Zhang,  Baochang*   and han, jungong and Liu, Jianzhuang}, year={2018}, IEEE Trans. Image processing.  }

@INPROCEEDINGS{Zhou2017ORN,
  title={Oriented Response Networks},
  author={Zhou, Yanzhao and Ye, Qixiang and Qiu, Qiang and Jiao, Jianbin},
  booktitle = {CVPR}
  year={2017},
}

