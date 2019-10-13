# Flowers17

Goal of this project is to create as tiny 'standard' CNN as possible, which reach high accuracy on small and demanding [flowers17 dataset](https://www.mozilla.org). 

## Architectures

For sake of experiment I was training four custom architectures instead of searching for best one. 

## Training

After training each architecture individually on 64x64 pictures, I chose best head shape and parameters tuple for each architecture and perform fine-tuning (train, freeze 'body', detach 'head', apply fresh one and train again for a while). It slightly decreased sizes and noticeably increased performance.

## VGG16

Additionally I performed fine-tuning on VGG16 pretrained on ImageNet dataset. 

## Result

Results describe shape of head, accuracy on testing set and rough number of parameters in network.

- MiniVGGNet:
    - before fine-tuning:
        - \[512\]: 79% (8.5 mil)
        - \[256\]: 77% (4.3 mil)
    - after fine-tuning:
        - \[512, 256\]: 83% (8.6 mil)
        - \[256\]: 84% (4.3 mil)
- LongNet:
    - before fine-tuning:
        - \[512, 256\]: 80% (8.7 mil)
        - \[256\]: 80% (4.4 mil)
    - after fine-tuning:
        - \[512, 256\]: 84% (8.7 mil)
        - \[256\]: 83% (4.4 mil)
- CascadeNet:
    - before fine-tuning:
        - \[512, 128\]: 84% (8.5 mil)
        - \[256\]: 77% (4.3 mil)
    - after fine-tuning:
        - \[512, 256\]: 84% (8.6 mil)
        - \[256\]: 85% (4.3 mil)
- WideNet:
    - before fine-tuning:
        - \[512, 256\]: 86% (27.4 mil)
        - \[256\]: 85% (14.2 mil)
    - after fine-tuning:
        - \[512, 256\]: 86% (27.4 mil)
        - \[256\]: 84% (14.2 mil)
- VGG16:
    - pictures 64x64:
        - \[256\]: 82% (15.2 mil)
    - pictures 128x128:
        - \[256\]: 92% (16.8 mil)