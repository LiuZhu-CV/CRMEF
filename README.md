# CRMEF
Embracing Compact and Robust Architectures for Multi-Exposure Image Fusion


## Fusion Reuslts and Chinese Version

The source images and fused results on three datasets are
provided in [link](https://drive.google.com/drive/folders/1KMVEM4oPOHgFCw4wKZfig2rEpLo0IESn)

中文版介绍提供在此链接 [link](https://arxiv.org/pdf/2308.03979.pdf)

Welcome all comparision and disscussion!
If you have any questions, please sending an email to "liuzhu_ssdut@foxmail.com"

## Preview of CRMEF
---
![preview](pics/workflow.png)
---
### General MEF
![General.png](pics%2FGeneral.png)

### Misalgined MEF
![Misaligned.png](pics%2FMisaligned.png)
## Set Up on Your Own Machine

### Virtual Environment

+ pytorch 1.2


### Test / Train
```shell
# Test: use given example and save fused color images to result/SICE
# If you want to test the custom data, please modify the file path in 'test.py'
python test_single.py

# the lightweight model
python test_single_lightweight.py

# if you want to test the alignment
cd DCNv2
sh make.sh
python test_align.py

# Train: 
python train.py
```

## Citation

If this work has been helpful to you, we would appreciate it if you could cite our paper! 

```
@article{liu2023embracing,
  title={Searching a Compact Architecture for Robust Multi-Exposure Image Fusion},
  author={Liu, Zhu and Liu, Jinyuan and Wu, Guanyao and Chen, Zihang and Fan, Xin and Liu, Risheng},
  journal={IEEE TCSVT},
  year={2024}
}
```