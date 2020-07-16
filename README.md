## Concept Learners for Generalizable Few-Shot Learning
Kaidi Cao*, Maria Brbić*, Jure Leskovec
_________________

This repo contains the reference source code of of the paper [Concept Learners for Generalizable Few-Shot Learning](https://arxiv.org/pdf/2007.07375.pdf) in PyTorch. Our codebase is developed based on the [benchmark implementation](https://github.com/wyharveychen/CloserLookFewShot) from paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ).  

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.5
- [anndata](https://icb-anndata.readthedocs-hosted.com/en/stable/anndata.AnnData.html)
- [scanpy](https://icb-scanpy.readthedocs-hosted.com/en/stable/)
- json
- [wandb](https://www.wandb.com/)

### Getting started
#### CUB
* Change directory to `./filelists/CUB`
* Run `source ./download_CUB.sh`

#### Tabula Muris
* Change directory to `./filelists/tabula_muris`
* Run `source ./download_TM.sh`

### Training

We provide an example here:

Run
```python ./train.py --dataset CUB --model Conv4NP --method comet --train_aug```

### Testing

We provide an example here:

Run
```python ./test.py --dataset CUB --model Conv4NP --method comet --train_aug```

### Reference

If you find our paper and repo useful, please consider cite

```
@inproceedings{
    cao2020concept,
    title={Concept Learners for Generalizable Few-ShotLearning},
    author={Cao, Kaidi and Brbi\'c, Maria and Leskovec, Jure},
    journal={arXiv},
    year={2020},
}
@inproceedings{
    chen2019closerfewshot,
    title={A Closer Look at Few-shot Classification},
    author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
    booktitle={International Conference on Learning Representations},
    year={2019}
}
```