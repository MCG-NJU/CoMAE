# CoMAE
[AAAI 2023] [CoMAE: Single Model Hybrid Pre-training on Small-Scale RGB-D Datasets](https://arxiv.org/abs/2302.06148)

![CoMAE](https://github.com/MCG-NJU/CoMAE/blob/main/framework.png?raw=true)

## Prepare Data
Baiduyun(https://pan.baidu.com/s/1LZIF1hlT3k0oX76Ttp660w) The extraction code is: g5vp

## Dependencies
* python 3.7.4
* torch 1.7.0
* torchvision 0.8.1
* timm 0.3.2
* numpy 1.17.2

## Pre-training
Note give your own data_path, output_dir and log_dir in command parameters.

 `python main_pretrain_cpc.py` or 
 
 `python -m torch.distributed.launch  --nproc_per_node NUM_GPU main_pretrain_cpc.py`
 
 Load CPC pretrained weights and `python main_pretrain_mm_mae.py` or 
 
 `python -m torch.distributed.launch  --nproc_per_node NUM_GPU main_pretrain_mm_mae.py`
 
 ## Fine-tuning and Evaluating
 Note give your own data_path, output_dir, log_dir and finetune in command parameters.
 
 `python main_finetune.py` or
 
 `python -m torch.distributed.launch  --nproc_per_node NUM_GPU main_finetune.py`

 ## Checkpoints on SUN RGB-D
cpc_stage1 [Google Drive](https://drive.google.com/file/d/1ncr6Kp3e3aVmdOu-0XrKF2K-ZZFdRF-Q/view?usp=drive_link)

mm_mae_stage2 [Google Drive](https://drive.google.com/file/d/1uq7xgwndsfk8ELFax9LlRNNomUnKXh2z/view?usp=drive_link)

finetune [Google Drive](https://drive.google.com/file/d/14tudeAIbZXOwfFDqMda2UG5K2SwHV7_u/view?usp=drive_link)
 
## Citation
Please cite the following paper if you feel this repository useful for your research.
```
@article{yang2023comae,
  title={CoMAE: Single Model Hybrid Pre-training on Small-Scale RGB-D Datasets},
  author={Yang, Jiange and Guo, Sheng and Wu, Gangshan and Wang, Limin},
  journal={arXiv preprint arXiv:2302.06148},
  year={2023}
}


@inproceedings{DBLP:conf/aaai/Yang0W023,
  author       = {Jiange Yang and
                  Sheng Guo and
                  Gangshan Wu and
                  Limin Wang},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {CoMAE: Single Model Hybrid Pre-training on Small-Scale {RGB-D} Datasets},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {3145--3154},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i3.25419},
  doi          = {10.1609/AAAI.V37I3.25419},
  timestamp    = {Mon, 04 Sep 2023 16:50:28 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/Yang0W023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
## Acknowledges
This repo contains modified codes from: [MAE](https://github.com/facebookresearch/mae).
