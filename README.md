# [NeurIPS 2024] Lightweight Frequency Masker for Cross-Domain Few-Shot Semantic Segmentation

This is the official implementation based on pytorch of the paper [Lightweight Frequency Masker for Cross-Domain Few-Shot Semantic Segmentation](https://arxiv.org/abs/2410.22135.pdf).[NeurIPS 2024]

Authors: [Jintao Tong](https://github.com/TungChintao), Yixiong Zou, Yuhua Li, Ruixuan Li

## Abstract

Cross-domain few-shot segmentation (CD-FSS) is proposed to first pre-train the model on a large-scale source-domain dataset, and then transfer the model to data-scarce target-domain datasets for pixel-level segmentation. The significant domain gap between the source and target datasets leads to a sharp decline in the performance of existing few-shot segmentation (FSS) methods in cross-domain scenarios. In this work, we discover an intriguing phenomenon: simply filtering different frequency components for target domains can lead to a significant performance improvement, sometimes even as high as 14% mIoU. Then, we delve into this phenomenon for an interpretation, and find such improvements stem from the reduced inter-channel correlation in feature maps, which benefits CD-FSS with enhanced robustness against domain gaps and larger activated regions for segmentation. Based on this, we propose a lightweight frequency masker, which further reduces channel correlations by an Amplitude-Phase Masker (APM) module and an Adaptive Channel Phase Attention (ACPA) module. Notably, APM introduces only 0.01% additional parameters but improves the average performance by over 10%, and ACPA imports only 2.5% parameters but further improves the performance by over 1.5%, which significantly surpasses the state-of-the-art CD-FSS methods.

<div align="center">
<img src="img/overview.png" width="100%" height="60%"/><br/>
</div>

## Dataset

You can follow [PATNet](https://github.com/slei109/PATNet) to prepare the source domain and target domain datasets.

## Source-Domain Training

Our module (APM, ACPA) is not involved in source-domain training; it is directly inserted during target-domain fine-tuning.

* **Pretrained model:** [ResNet-50](https://drive.google.com/file/d/11yONyypvBEYZEh9NIOJBGMdiLLAgsMgj/view?usp=sharing) 

* **Base model trained in original [HSNet](https://github.com/juhongm999/hsnet) in the manner of [PATNet](https://github.com/slei109/PATNet)**

* Our checkpoint of HSNet trained on the source domain is available here: [[Google Drive](https://drive.google.com/file/d/1qvsMneFbWyZoaux45TiD-Rvucew1EK4_/view?usp=drive_link)].

## Target-Domain Finetuning/Target-Domain Testing

We provide an example of integrating our lightweight frequency masker into HSNet. Our module is model-agnostic and can be applied to other models as well.

## Usage

We provide an example of using APM (Amplitude-Phase Masker) and ACPA (Adaptive Channel Phase Attention), which are integrated into the model to operate on intermediate features.

```python
import torch
from .freq_masker import MaskModule
from .phase_attn import PhaseAttention

bsz, c, h, w = 3, 2048, 13, 13
immediate_feature = torch.rand(bsz, c, h, w)

# APM-S
apm_s = MaskModule([1,1,h,w])
# APM-M
apm_m = MaskModule([1,c,h,w])
# ACPA
acpa = PhaseAttention(c)

enhanced_feature = apm_s(immediate_feature) # apm_m(immedidate_feature)
final_feature = acpa(enhanced_feature)
```

## Bibtex

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{tong2024lightweight,
	title={Lightweight frequency masker for cross-domain few-shot semantic segmentation},
	author={Tong, Jintao and Zou, Yixiong and Li, Yuhua and Li, Ruixuan},
	journal={Advances in Neural Information Processing Systems},
	volume={37},
	pages={96728--96749},
	year={2024}
}
```

## Acknowledgement

The implementation is based on [HSNet](https://github.com/juhongm999/hsnet) and [PATNet](https://github.com/slei109/PATNet)
