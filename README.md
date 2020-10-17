# MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/HKUST.jpg" width="12%">

This is the implementation of the **EMNLP 2020** paper:

**MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems**. [**Zhaojiang Lin**](https://zlinao.github.io/), [**Andrea Madotto**](https://andreamad8.github.io), [**Genta Indra Winata**](https://gentawinata.com), Pascale Fung  [[PDF]](https://arxiv.org/pdf/2009.12005.pdf)


## Citation:
If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@article{lin2020mintl,
    title={MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems},
    author={Zhaojiang Lin and Andrea Madotto and Genta Indra Winata and Pascale Fung},
    journal={arXiv preprint arXiv:2009.12005},
    year={2020}
}
</pre>


## Abstract:
In this paper, we propose Minimalist Transfer Learning (MinTL) to simplify the system design process of task-oriented dialogue systems and alleviate the over-dependency on annotated data. MinTL is a simple yet effective transfer learning framework, which allows us to plug-and-play pre-trained seq2seq models, and jointly learn dialogue state tracking and dialogue response generation. Unlike previous approaches, which use a copy mechanism to "carryover" the old dialogue states to the new one, we introduce Levenshtein belief spans (Lev), that allows efficient dialogue state tracking with a minimal generation length. We instantiate our learning framework with two pretrained backbones: T5 (Raffel et al., 2019) and BART (Lewis et al., 2019), and evaluate them on MultiWOZ. Extensive experiments demonstrate that: 1) our systems establish new state-of-the-art results on end-to-end response generation, 2) MinTL-based systems are more robust than baseline methods in the low resource setting, and they achieve competitive results with only 20% training data, and 3) Lev greatly improves the inference efficiency.


## Dependency
Check the packages needed or simply run the command
```console
❱❱❱ pip install -r requirements.txt
```

## Experiments Setup
We used the preprocess script from [**DAMD**](https://gitlab.com/ucdavisnlp/damd-multiwoz).
Please check setup.sh for data preprocessing.

## Experiments
**T5 End2End**
```console
❱❱❱ python train.py --mode train --context_window 2 --pretrained_checkpoint t5-small --cfg seed=557 batch_size=32
```
**T5 DST**
```console
❱❱❱ python DST.py --mode train --context_window 3 --cfg seed=557 batch_size=32
```

**BART End2End**
```console
❱❱❱ python train.py --mode train --context_window 2 --pretrained_checkpoint bart-large-cnn --gradient_accumulation_steps 8 --lr 3e-5 --back_bone bart --cfg seed=557 batch_size=8
```
**BART DST**
```console
❱❱❱ python DST.py --mode train --context_window 3 --gradient_accumulation_steps 10 --pretrained_checkpoint bart-large-cnn --back_bone bart --lr 1e-5 --cfg seed=557 batch_size=4
```

check run.py for more information.

