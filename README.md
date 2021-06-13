# Team sOcCeR's formular recognizer(OCR)

## [목차]

* [수식 인식기 프로젝트 소개](#수식-인식기-프로젝트-소개)
* [Environment](#Environment)
    * [Hardware](#Hardware)
    * [Software](#Software)
    * [Dependencies](#Dependencies)
* [Usage](#usage)
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

## [Environment]

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

## Usage

모델을 사용하기 위해서는 ~~~ 를 실행시킵니다.
```bash
$ python train.py --c your_model.yaml
```

총 3가지 모델을 선택할 수 있습니다.

- **Attention(ASTER)**
- **SATRN**
- **ViT**




## File Structure
- ___코드 정리 후 수정___

```python
p4-fr-soccer/code/
│
├── configs
├── datatools
├── log
├── network
├── submit
├── checkpoint.py
├── dataset.py
├── dataset_ViT.py
├── submit
└── data_utils.py
```
___코드 정리 후 수정___

## Models


### ASTER

### SATRN

### ViT






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

### Dataset

- 
