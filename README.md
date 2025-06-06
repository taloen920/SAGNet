# Referring Remote Sensing Image Segmentation Method based on Scene-Aware Guided Network Mode
![SAGNet](C:\Users\TK\Desktop\论文图列\最终\SAGNet.png)

## Setting Up
### Preliminaries
The code has been verified to work with PyTorch v1.7.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.
### Package Dependencies
1. Create a new Conda environment with Python 3.7 then activate it:
```shell
conda create -n SAGNet python==3.7
conda activate SAGNet
```

2. Install PyTorch v1.7.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```

### The Initialization Weights for Training
1. Create the `./pretrained_weights` directory where we will be storing the weights.
```shell
mkdir ./pretrained_weights
```
2. Download [pre-trained classification weights of
   the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
   and put the `pth` file in `./pretrained_weights`.
   These weights are needed for training to initialize the visual encoder.
3. Download [BERT weights from HuggingFace’s Transformer library](https://huggingface.co/google-bert/bert-base-uncased), 
   and put it in the root directory. 

## Datasets

We perform the experiments on three dataset including [RefSegRS](https://github.com/zhu-xlab/rrsis) ,[RRSIS-D](https://github.com/Lsan2401/RMSIN) and our proposed dataset LandRef.It can be downloaded from 


## Training
We use `DistributedDataParallel` from PyTorch. To run on 2 GPUs (with IDs 0 and 1) on a single node:
```shell
sh train.sh
```
## Testing
```shell
sh test.sh
```
## Acknowledgements
Code in this repository is built on [LAVT](https://github.com/yz93/LAVT-RIS). We'd like to thank the authors for open sourcing their project.
