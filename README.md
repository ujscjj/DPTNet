# DPTNet

A PyTorch implementation of dual-path transformer network (DPTNet) based speech separation on wsj0-2mix described in the  paper  <a href="https://arxiv.org/abs/2007.13975">"Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation"</a>, which has been accepted by Interspeech2020.  



This implementation is based on <a href="https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation">DPRNN</a>, thanks Yi Luo and ShiZiqiang for sharing.

File description:

> optimizer.py：                               a simple wrapper class for learning rate scheduling
>
> transformer_improved.py：       a PyTorch implementation of the improved transformer in the paper
>
> models.py：                                   use the "SingleTransformer" in our model.py to replace the "SingleRNN" in <a href="https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation/blob/master/models.py">shi's model.py</a>

You only need to perform the above replacement based on the codes in <a href="https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation">DPRNN</a>, then you can implement the method in <a href="https://arxiv.org/abs/2007.13975">dual-path transformer network (DPTNet)</a>.

We obtain SDR 20.6 dB on wsj0-2mix and 16.8 dB on LS-2mix dataset.

##### I will organize the complete codes and upload them when I am free.

#### References

<a href="https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation">https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation</a>

<a href="https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/optimizer.py">https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/optimizer.py</a>

<a href="https://github.com/pytorch/pytorch/blob/eace0533985641d9c2f36e43e3de694aca886bd9/torch/nn/modules/transformer.py">https://github.com/pytorch/pytorch/blob/eace0533985641d9c2f36e43e3de694aca886bd9/torch/nn/modules/transformer.py</a>

