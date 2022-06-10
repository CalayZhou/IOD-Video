# Installation

>**Our experimental environment:** 
>
> Ubuntu 18.04.5  Python 3.9.6, PyTorch 1.8.1.
> 
> During training we use 2 NVIDIA GeForce RTX 3090 with cuda11.1 (About 48G memory needed).


## 1.Basic Usage
1. Create a new conda environment and activate the environment.

   ~~~powershell
   conda create --n IOD python=3.9
   conda activate IOD
   ~~~
   
   
2. Install pytorch1.8.1:

   ~~~powershell
   conda install pytorch==1.8.1 torchvision cudatoolkit -c pytorch
   ~~~
   
   The torch version is not supposed to be 1.8.1, it can be changed to any other version, only the cuda version is required to be same with `CUDA environment(/usr/local/cuda)` for [Pytorch Correlation extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
<br>
   
3. Clone this repo (${MOC_ROOT} is the path to clone):

   ~~~powershell
   git clone https://github.com/CalayZhou/IOD-Video.git 
   ~~~


4. Install the requirements

   ~~~powershell
   pip install -r pip-list.txt
   ~~~


## 2.Additional Usage
1. cuda_shift for TIN
   ~~~powershell
   cd ./network/twod_models/cuda_shift
   bash make.sh
   ~~~

   You can refer to [TIN](https://github.com/deepcs233/TIN) for detail.


2. Pytorch-Correlation-extension for MSNet
   
   ~~~powershell 
   git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
   python setup.py install 
   ~~~
   
   The cuda version of torch should be same with `CUDA environment( /usr/local/cuda)`, otherwise it may cause issue as mentioned in [#80](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues/80). If GPUs are NVIDIA 30 series, the version can be changed to the branch [fix_1.7](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues/52). You can refer to [MSNet](https://github.com/arunos728/MotionSqueeze) and [Pytorch Correlation extension](https://github.com/ClementPinard/Pytorch-Correlation-extension) for more information.


The basic usage can satisfy most spatio-temporal backbones except TIN and MSNet. You can comment `TINresnet` and `MSresnet` in `STA_Framework.py` for fast implementation.


