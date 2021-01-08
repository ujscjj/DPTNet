# DPTNet

A PyTorch implementation of dual-path transformer network (DPTNet) based speech separation on wsj0-2mix described in the  paper  <a href="https://arxiv.org/abs/2007.13975">"Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation"</a>, which has been accepted by Interspeech2020.  



This implementation is based on <a href="https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation">DPRNN</a>, thanks Yi Luo and ShiZiqiang for sharing.

File description:

> optimizer_dptnet.py：                               a simple wrapper class for learning rate scheduling
>
> transformer_improved.py：                               a PyTorch implementation of the improved transformer in the paper
>
> dpt_net.py：                               where you can start

We obtain SDR 20.6 dB on wsj0-2mix and 16.8 dB on LS-2mix dataset.

## Train

```
CUDA_VISIBLE_DEVICES=0 python train_and_eval.py --train_dir /my/wsj2-mix/path/min/tr --valid_dir /my/wsj2-mix/path/min/cv
```

Continue training without eval:

```
CUDA_VISIBLE_DEVICES=0 python train_and_eval.py --mode train --continue_from exp/temp_ctd/epoch14.pth.tar --start_epoch 14 --warmup 0 --save_folder exp/tmp_ctd
```

Train a faster training model

```
CUDA_VISIBLE_DEVICES=1 python train_and_eval.py --print_freq 40 --W 16 --D 2 --save_folder exp/smaller_spectrogram
```

Eval only

```
CUDA_VISIBLE_DEVICES=1 python train_and_eval.py --mode eval --continue_from exp/temp_ctd2/epoch8.pth.tar --save_folder exp/tmp_ctd2_eval
```


### References

<a href="https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation">https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation</a>

<a href="https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/optimizer.py">https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/optimizer.py</a>

<a href="https://github.com/pytorch/pytorch/blob/eace0533985641d9c2f36e43e3de694aca886bd9/torch/nn/modules/transformer.py">https://github.com/pytorch/pytorch/blob/eace0533985641d9c2f36e43e3de694aca886bd9/torch/nn/modules/transformer.py</a>

### Requirements

Working in pytorch 1.8, cuda 11.1.
Default training model is working on a gpu with 16+ GB of memory.
Faster-training model is working on a gpu with only 2.5 GB of memory.
