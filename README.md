# CoMAE
[AAAI 2023] [CoMAE: Single Model Hybrid Pre-training on Small-Scale RGB-D Datasets](https://arxiv.org/abs/2302.06148)

## Prepare Data
Baiduyun(https://pan.baidu.com/s/1LZIF1hlT3k0oX76Ttp660w) The extraction code is: g5vp

## Dependencies
* python 3.7.4
* torch 1.7.0
* torchvision 0.8.1
* timm 0.3.2
* numpy 1.17.2

## Pretrain
Note give your own data_path, output_dir and log_dir in command parameters.

 `python main_pretrain_cpc.py`
 
 Load CPC pretrained weights and `python main_pretrain_mm_mae.py`
 
 ## Finetune and eval
 Note give your own data_path, output_dir, log_dir and finetune in command parameters.
 
 `python main_finetune.py`
 
## Citation
Please cite the following paper if you feel this repository useful for your research.
```
@inproceedings{yang2023comae,
  title={CoMAE: Single Model Hybrid Pre-training on Small-Scale RGB-D Datasets},
  author={Yang, Jiange and Guo, Sheng and Wu, Gangshan and Wang, Limin},
  year={2023}
}
```
## Acknowledges
This repo contains modified codes from: [MAE](https://github.com/facebookresearch/mae).
