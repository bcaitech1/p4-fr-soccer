# Team sOcCeR's formular recognizer(OCR)

## 목차

* [수식 인식기 프로젝트 소개](#수식-인식기-프로젝트-소개)
* [Environment](#Environment)
    * [Hardware](#Hardware)
    * [Software](#Software)
    * [Dependencies](#Dependencies)
* [Usage](#usage)
* [Issue](#Issue)
* [File Structure](#File-structure)
* [Models](#Models)
    * [ASTER](#ASTER)
    * [SATRN](#SATRN)
    * [ViT](#ViT)
* [Contributors](#contributors)
* [Reference](#reference)
    * [Papers](#papers)


## 프로젝트 기간
- 2021년 05월 24일 ~ 2021년 06월 15일

## 프로젝트 overview
![last structure](https://user-images.githubusercontent.com/52662993/121977545-38303e00-cdc1-11eb-8b74-1243c8dd0c49.PNG)



## 수식 인식기 프로젝트 소개
![formular](https://user-images.githubusercontent.com/52662993/121814480-48a2c480-ccac-11eb-8201-96eed32a245b.png)
- **수식인식기 프로젝트** 는 수식이 적혀있는 이미지를 LaTex표현으로 바꾸는 Image to text 테스크 입니다.
- 수식이 적혀있는 **손글씨 5만장, 인쇄물 5만장**의 데이터로 학습을 진행하고 **1만2천장**의 수식 이미지를 LaTex로 변환하는 과제입니다. 

## 평가 방법
- 0.9 * **문장 단위 정확도** + 0.1*(1 - **단어 오류율**)
- **문장 단위 정확도**(Sentence Accuracy)(%): 정답과 정확하게 일치하는 문장 갯수 / 전체 문장 갯수
 <img width="717" alt="ACC" src="https://user-images.githubusercontent.com/52662993/121814386-c3b7ab00-ccab-11eb-977e-116ef05a1502.png">
- **단어 오류율**(Word Error Rate)(%): 삽입, 삭제, 대체가 필요한 횟수 / 전체 단어 수
<img width="499" alt="WER" src="https://user-images.githubusercontent.com/52662993/121814389-c87c5f00-ccab-11eb-91a4-65e95aef6b21.png">

## Environment

### Hardware

- CPU: Xeon Gold 5120
- GPU: Tesla V100 32GB
- Mem: > 90GB
- Data is stored in remote server storage.

### Software

- System: Ubuntu 18.04.4 LTS with Linux 4.4.0-210-generic kernel.
- Python: 3.7 distributed by Anaconda.
- CUDA: 10.1
- Pytorch: 1.4.0

### Dependencies

- scikit_image==0.14.1
- opencv_python==3.4.4.19
- tqdm==4.28.1
- torch==1.4.0
- scipy==1.2.0
- numpy==1.15.4
- torchvision==0.2.1
- Pillow==8.1.1
- tensorboardX==1.5
- editdistance==0.5.3

```python
$ pip install -r requirements.txt
```

### Attention, SATRN
```bash
$ python train.py --c {your_model}.yaml
```
총 2가지 모델을 선택할 수 있습니다.
- **Attention(ASTER)**
- **SATRN**
### Vit
```bash
$ python train_ViT.py
```
### Swin
```bash
$ python train_swin.py
```

## Dataset
학습이미지 예시:\
![image](https://user-images.githubusercontent.com/52662993/121864735-30788700-cd38-11eb-9519-26288b7f0d88.png)

Ground Truth:\
x = \frac { - b \pm \sqrt { b ^ 2 - 4 a c } } { 2 a }  \ { \text { when } } \ {a x ^ 2 + b x + c = 0}



## File Structure

```python
p4-fr-hatting-day/code/
│
├── configs
│    ├── Attention.yaml
│    ├── EFFICIENT_SATRN.yaml
│    ├── EFFICIENT_SATRNv6.yaml
│    ├── ...
│    └── swin.py
├── datatools
│    ├── extract_tokens.py
│    ├── parse_upstage.py
│    └── train_test_split.py
├── network
│    ├── Attention.py
│    ├── EFFICIENT_SATRN.py
│    ├── SATRN_extension.py
│    ├── ...
│    └── swin.py
├── submit
├── checkpoint.py
├── dataset.py
├── dataset_ViT.py
├── dataset_Swin.py
├── floags.py
├── inference.py
├── inference_ensemble.py
├── metrics.py
├── requirements.txt
├── requirements_2.txt
├── scheduler.py
├── submission.txt
├── train.pt
├── train_ViT.py
├── train_swin.py
└── utils.py
```

## Models


### ASTER
- CNN과 LSTM으로 구성된 Encoder와 Encoder output과 전 LSTM의 hidden state를 Attention하는 모델입니다.
- Scene text recognition의 기초 모델입니다.
- BLSTM의 hidden state를 더하여 디코더로 넘겨주었습니다.
- CNN backbone: EfficientNet V2
<p align="center">
<img width="406" alt="스크린샷 2021-06-14 오후 5 19 14" src="https://user-images.githubusercontent.com/52662993/121861277-a975df80-cd34-11eb-9e64-85dd16b2c8e3.png">
</p>

### SATRN
- ASTER와 마찬가지로 Encoder, Decoder로 구성된 모델입니다.
- 이미지의 수평, 수직정보의 중요도를 학습하는 A2DPE, 문자 주변 공간정보를 학습하는 Locality-aware feedforward가 특징인 모델입니다.
- Multi head attention 진행시 [Residual attention](https://arxiv.org/pdf/2012.11747.pdf)을 적용하여 성능 개선
- Weight initialize는 [RealFormer논문](https://arxiv.org/pdf/2012.11747.pdf)을 참고하였습니다.
- CNN backbone: ResnetRS152, EfficientNet v2를 사용하였습니다.
<p align="center">
<img width="481" alt="스크린샷 2021-06-14 오후 5 23 34" src="https://user-images.githubusercontent.com/52662993/121893056-7f361900-cd58-11eb-894e-69806f4e077e.png">
</p>




### ViT
- 이미지를 patch로 나누어 하나의 시퀀스로 취급하여 transformer를 통해 학습하는 모델입니다.
- 
<p align="center">
<img width="481" alt="스크린샷 2021-06-14 오후 5 23 34" src="https://user-images.githubusercontent.com/52662993/121861936-49336d80-cd35-11eb-85c8-875409a1df63.png">
</p>

### Swin
<p align="center">
<img width="943" alt="스크린샷 2021-06-14 오후 5 24 11" src="https://user-images.githubusercontent.com/52662993/121861984-57818980-cd35-11eb-97e3-c1cc0a33d3f6.png">
</p>

## Contributors



- **이동빈** ([Dongbin-Lee-git](https://github.com/Dongbin-Lee-git))
- **이근재** ([GJ Lee](https://github.com/lgj9172))
- **이정환** ([JeonghwanLee1](https://github.com/JeonghwanLee1))
- **조영민** ([joqjoq966](https://github.com/joqjoq966))
- **김수호** ([Sooho-Kim](https://github.com/Sooho-Kim))
- **신문종** ([moon-jong](https://github.com/moon-jong))



## [Reference]



### Papers
- [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/pdf/1910.04396v1.pdf)
- [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](http://122.205.5.5:8071/UpLoadFiles/Papers/ASTER_PAMI18.pdf)
- [RealFormer: Transformer Likes Residual Attention](https://arxiv.org/pdf/2012.11747.pdf)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929v2.pdf)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298)
- [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)
- [Repulsive Attention:Rethinking Multi-head Attention as Bayesian Inference](https://arxiv.org/pdf/2009.09364.pdf)
- [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Contrastive Learning for Unpaired Image-to-Image Translation](https://arxiv.org/pdf/2007.15651.pdf)
- [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171)

### Supported Data
- [Aida](https://www.kaggle.com/aidapearson/ocr-data)
- [CHROME](https://www.isical.ac.in/~crohme/)
- [IM2LATEX](http://lstm.seas.harvard.edu/latex/)
- [Upstage](https://www.upstage.ai/)
