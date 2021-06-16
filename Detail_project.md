# 대회 개요


## 프로젝트 기간
- 2021년 05월 24일 ~ 2021년 06월 15일

## 프로젝트 목표

- 손글씨와 프린트된 수식 이미지를 latex 포맷의 텍스트로 변환하는 모델을 개발합니다.

## 입력 및 출력

![Project%20Detail/Untitled.png](Project%20Detail/Untitled.png)

### 입력

- 수식이 표현된 이미지

### 출력

- LaTeX 포맷의 텍스트

## 데이터

### 학습 데이터

- 10만장(출력물 5만장, 손글씨 5만장)

### 테스트 데이터

- 1.2만장(출력물 0.6만장, 손글씨 0.6만장)

## 평가방법

- 0.9 * **문장 단위 정확도** + 0.1*(1 - **단어 오류율**)
- **문장 단위 정확도**(Sentence Accuracy)(%): 정답과 정확하게 일치하는 문장 갯수 / 전체 문장 갯수
- **단어 오류율**(Word Error Rate)(%): 삽입, 삭제, 대체가 필요한 횟수 / 전체 단어 수

## 최종 결과

- *Public Score : 0.8204 (2nd)*
- *Private Score : 0.5956 (3rd)*


## Log

[stage-4-soccer](https://wandb.ai/stage-4-soccer/OCR)

# 프로젝트 방향성


세부적인 Hyper Parameter조절보다 모델의 구조를 통해 성능을 끌어올리는 것으로 프로젝트의 방향성을 정했습니다.

다양한 논문을 읽어보면서 아이디어를 직접 저희의 모델에 적용해보는 방식을 활용했습니다.

적용하기 전에는 근거에 의해서 모델의 문제를 해결하려고 노력했습니다.

최초 3개의 이상의 모델을 앙상블할 계획으로 다양한 모델을 실험하려고 노력했습니다.

# Environment


We trained models on our lab's Linux cluster. The environment listed below reflects a typical software / hardware configuration in this cluster.

### **Hardware:**

- CPU: Xeon Gold 5120
- GPU: Tesla V100 32GB
- Mem: > 90GB
- Data is stored in remote server storage.

### **Software:**

- System: Ubuntu 18.04.4 LTS with Linux 4.4.0-210-generic kernel.
- Python: 3.7 distributed by Anaconda.
- CUDA: 10.1
- Pytorch: 1.4.0

# Pipeline


### All of what we experiments

![image](https://user-images.githubusercontent.com/57521132/122161850-6be89200-cead-11eb-99c4-a5a3407c42bf.png)

### Flow of what we applied

![Project%20Detail/Untitled%202.png](Project%20Detail/Untitled%202.png)

## 데이터


### EDA

- 데이터 셋 특징
    - 데이터의 비율에 대한 EDA 결과(df : 원래 주어진 데이터, info : 세로 이미지 rotate한 뒤 결과)
<p align="center">
<img width = "481"src="Project%20Detail/Untitled%203.png">
</p>

- 해당 EDA를 통해 4:1 비율로 이미지를 넣어주는 것이 좋다고 생각했습니다. 확실하게 정사각형 형태로 넣어주는 것보다 좋은 성능을 보였습니다. 64:256비율과 128:512 중에 좀 더 큰 이미지를 쓰는 것이 좋다고 판단했습니다.
- EDA를 하면서 세로의 형태의 이미지가 잘못 학습을 할 수 있을 것 같다고 생각 → 추후 가로 형태 데이터 셋 제작
- 손 글씨의 경우, 검은 색 볼펜 이외에도 다양한 색상의 볼펜, 형광펜을 사용하는 경우가 있었습니다 → gray scale로 변경해서 글씨에 집중하도록 수정
- 뒤집었을 때, 출력하는 결과가 달라지는 경우도 존재했습니다. 0+1을 뒤집었을 때, 1+0으로 결과를 prediction하게 되었을 때, loss가 크게 형성될 것이라고 생각했습니다. → rotate는 적용하지 않는 것이 좋다고 판단. 실제로 rotate를 180도 적용한 경우 validation 기준으로 0.05 가량 차이 존재.

    ![Project%20Detail/Untitled%204.png](Project%20Detail/Untitled%204.png)

실제 인퍼런스 결과 → 하지만 이를 변경해주기는 어렵다고 판단. Data noise로 생각하기로 했습니다.

<p align="center">
<img width = "481"src="Project%20Detail/Untitled%205.png">
</p>


- width보다 height가 긴 이미지 약 2400장 중 1700장은 세로 형태 이미지 → 가로 형태로 임의로 90도 회전시키고 rotation 180을 줬으나 성능 0.04가량 하락 → 1700장에 대해 올바르게 학습하도록 dataset set 수정 →  public에서 0.01정도 성능 향상

  ![Project%20Detail/Untitled%206.png](Project%20Detail/Untitled%206.png)

- resize하는 연산이 학습 속도에 영향을 준다고 판단 → 최종으로 사용할 4:1 형태인 128:512 사이즈로 resize → 학습 속도 향상

### 데이터 전처리

- CycleGAN as a Denoising Engine for OCR Images ([https://pub.towardsai.net/cyclegan-as-a-denoising-engine-for-ocr-images-8d2a4988f769](https://pub.towardsai.net/cyclegan-as-a-denoising-engine-for-ocr-images-8d2a4988f769))

    위의 링크 글을 보고 이번 프로젝트에도 적용할 수 있지 않을까 하는 생각이 들었다.

    - CycleGAN을 학습하기 위해서 target Data가 필요했고, 정제하는 과정을 진행했다.
    - opencv를 사용하여 실험을 진행했고, 처음에는 Binary+OTSU thresholding을 통해 black, white의 경계를 분리시켜주어 수식만 뽑히도록 진행했지만, 그림자에 의한 음영이나 흐릿한 이미지들은 잘 분리해내지 못했다.
    - 이를 해결하기 위해 CLAHE를 사용하여 색상의 대조를 뚜렷하게 만들어주었고, 약간의 효과가 있었다.
    - 큰 효과가 없었기에 CLAHE의 과정을 깊게 보았고, train dataset의 길이에 맞추어 동적으로 Grid size가 적용되도록 수정하여 조금더 개선했다.
    - 하지만 음영이 어느정도 강하게 들어있는 data들에 대해서는 문제점이 계속 발생했다.

        ![Project%20Detail/Untitled%207.png](Project%20Detail/Untitled%207.png)

    - 이미지의 한쪽 부분들이 어두운 음영을 가지고 있을때에 위와 같은 현상이 더 발생하고 threshold의 과정에서 이러한 문제점이 생길 것 같다는 판단 → image size에 대해 동적으로 white padding을 적용해 주면 어떨까? → 아래의 결과

        ![Project%20Detail/Untitled%208.png](Project%20Detail/Untitled%208.png)

    - 위 cleaning data를 B, origin data를 A로 주어 Cycle GAN, CUT(Contrastive Unpaired Translation), FastCUT을 학습
        - CycleGAN result
            - v1 [https://github.com/bcaitech1/p4-fr-soccer/discussions/11](https://github.com/bcaitech1/p4-fr-soccer/discussions/11)
            - v2 [https://github.com/bcaitech1/p4-fr-soccer/discussions/16](https://github.com/bcaitech1/p4-fr-soccer/discussions/16)
        - CycleGAN vs CUT result
            - [https://github.com/bcaitech1/p4-fr-soccer/discussions/24](https://github.com/bcaitech1/p4-fr-soccer/discussions/24)
        - FastCUT result
            - [https://github.com/bcaitech1/p4-fr-soccer/discussions/33](https://github.com/bcaitech1/p4-fr-soccer/discussions/33)
    - GAN을 통해 변환한 Data를 input으로 OCR 학습을 진행 → 성능 하락
        - 원인이 뭘까?
            - GAN이 모든 Train Data에 대해 Cleaning작업을 완벽하게 적용하지 못했다.
            - GrayScale의 Data들이 이미 다양한 augmentation(blur, contrast, etc.)을 내포하고 있기에 오히려 cleaning작업이 불필요하게 작용했다.
        - GAN에 대한 더 이상의 실험 중단.
- Image preprocessing (opencv, Albumentations)
    - GAN으로 실패했으나, cv2와 Albumentations 통해 학습 시도.
    - 명암 제거하기 위해서 cv2.dilate 함수를 이용한 뒤 medianBlur 처리한 것을 원본 이미지에서 빼주는 작업을 진행하고 normalize를 통해서 최대한 명암을 제거했습니다.
    - 명암 조절한 이미지를 Albumentations를 적용했으나, inference 과정에서 라이브러리가 맞지 않는 문제 때문에 사용하지 않았습니다.
    - 최종적으로 위에서 적용하려고 했던 것들은 생각보다 성능이 좋지 않아서 사용하지 않았습니다.
    - 왜 그렇게 되었을까?
        - 모든 이미지에 대해 깔끔한 형태로 만들어주는 것은 어려웠다고 생각합니다. 최대한 많은 이미지를 확인하고 싶었으나, 다 확인하기 어려웠습니다.
- 참고자료

## 모델링

### ASTER

- BLSTM 추가

    기존 베이스라인에 Dense Block을 거친 후 Feature map을 디코더 LSTM블록

    기존 방식:  Conv net을 거친 feature map을 곧바로 Decoder의 hidden state와 attention
    <p align="center">
    <img width="300" height="400" src="Project%20Detail/_2021-06-13__3.51.22.png">
    </p>

    BLSTM 추가: Feature map을 BLSTM을 거쳐 디코더의 hidden state와 attention
    <p align="center">
    <img width="300" height="400" src="Project%20Detail/_2021-06-13__3.53.33.png">
    </p>
    

    1. ConvNet을 거치며 겹치지 않은 receptive field를 BLSTM을 거치며 보완할 수 있을거라 생각했습니다
    2. 이미지를 LaTeX로 바꿀 때 한방향으로만 이미지를 읽는다면 루트, 분수를 잘 표현하지 못할것이라 생각하여 양방향 LSTM을 거치며 수식 이미지의 문맥을 인코더 feature로써 디코더에 전달하였습니다

    결론: 기존 LSTM을 추가하지 않았을 때 0.5757 → BLSTM 추가 0.6051로 상승하였습니다.

- Backbone 교체

    단순 CNN이었던 encoder layer를 CNN 계열 SOTA 모델인 EfficientNet v2로 교체하였습니다.
    <p align="center">
    <img height="400" src="Project%20Detail/_2021-06-13__4.20.32.png">
    </p>

    기존 BLSTM을 적용하여 0.6051 val score에서 0.6465로 4% 성능 향상

### SATRN

- Weight Initialize
    1. Kaiming(Default)
        - Custom transformer를 사용 → Linear Layer의 Initialize는 어떻게 되어 있을까? → pytorch 1.4.0 github Linear.py ([https://github.com/pytorch/pytorch/blob/v1.4/torch/nn/modules/linear.py](https://github.com/pytorch/pytorch/blob/v1.4/torch/nn/modules/linear.py)) initialize가 Kaiming으로 적용. → initialize를 바꾸어 성능향상을 끌어올릴 수 있지 않을까?
    2. Xavier
        - pytorch 1.4.0 github의 transformer([https://github.com/pytorch/pytorch/blob/v1.4/torch/nn/modules/transformer.py](https://github.com/pytorch/pytorch/blob/v1.4/torch/nn/modules/transformer.py))의 initial weight가 xavier_uniform으로 되어있고, 다양한 transformer참고 자료에도 xavier initialize를 사용하는 점을 참고하여 실험을 진행.

            ![Project%20Detail/Untitled%209.png](Project%20Detail/Untitled%209.png)

        - 기존(검은색)에 비해 크게 낮은 성능을 보여주었고, 이를 보완해주기 위해 LR을 높여주거나 Corssentropy with ignore_index를 적용해도 저조한 성능을 보였다.
        - CNN block에서 여러 RELU를 거친 Feature가 transformer input으로 들어가게 되고,  Xavier initialize의 RELU에서의 문제점을 Kaiming initialize에서 개선해주는 점들이 맞물려 Xavier가 낮은 성능을 보이는 것 같다.
    3. Custom
        - RealFormer(Residual Attention)을 적용하다가 다른 Initialize를 보게되었고, 이를 model에 적용. ([https://github.com/cloneofsimo/RealFormer-pytorch/blob/main/models.py](https://github.com/cloneofsimo/RealFormer-pytorch/blob/main/models.py))

            ![Project%20Detail/Untitled%2010.png](Project%20Detail/Untitled%2010.png)

        - Train/Val Score/Loss가 모두 적용전보다 향상된 성능을 보였고, 최종 model에도 적용.
- CNN Block

    원 논문 구조에서 SATRN의 CNN Block은 Shallow CNN으로 구성이 되어있으며, 다른 deep한 Imagenet들에 비해 얕은 layer로 질감, 패턴들만 추출하여 Transformer의 input으로 적용된다. 

    [https://github.com/bcaitech1/p4-fr-soccer/discussions/34](https://github.com/bcaitech1/p4-fr-soccer/discussions/34)

    → Timm라이브러리를 사용하여 deep하지만, pretrain되어진 weight들이 이를 보완해 줄 수 있지않을까? → pretrained Imagenet사용 → 성능 대폭 향상 

    1. DeepCNN300(baseline)
        - Baseline으로 주어진 CNN Block이다. Densenet을 Customize하게 구성되어져있었으며 output channel이 300 dimension을 만족하도록 구성이 되어졌다.

    2. DenseNet
        - 위 DeepCNN300이 custom Densenet으로 구성이 되었기에 Pretrained를 사용하면 어떨까? 하는 생각으로 실험이 진행되었다.

            ![Project%20Detail/Untitled%2011.png](Project%20Detail/Untitled%2011.png)

        - Pretrain Model을 사용하는 것이 더 좋은 성능을 보였다.

    3. EfficientNetv2
        - EfficientNetv2의 성능이 CNN에서의 SOTA로 알려져 있어 사용하게 되었다.
        - timm 라이브러리에 적용되어 있는 Pretrained Model을 사용하였다.
            <p align="center">
            <img height="400" src="Project%20Detail/Untitled%2012.png">
            </p>

        - 학습 초기에는 Pretrained의 영향으로 인해 DeepCNN300보다 높은 성능을 보여주었다.
        - 학습 중간에는 갑자기 성능이 하락하는 등 불안정한 성능을 보여주었고, 최종 성능은 DeepCNN300 보다 낮았다.

            ![Project%20Detail/Untitled%2013.png](Project%20Detail/Untitled%2013.png)

        - 기존보다 성능은 감소하였으나, 학습이 전반적으로 잘 진행되는 추이를 보여주었고, 하이퍼파라미터를 변경하지 않고 단순히 네트워크만 변경해준 결과이기 때문에 모델의 다양성을 위해 EfficientNetv2을 계속 사용하였다.

    4. ResnetRS
        - Efficientnet의 학습모습과 Densenet의 학습 모습을 보고, 많이 Deep한 모델보다 원초적인 구성으로 진행된 Imagenet이 좋을 것 같다는 생각으로 Timm Library에서 모델을 찾기 시작했다.
        - Resnet부터 시작해서 보던 중, 올해 초 발표되어진 ResnetRS를 보았고 논문에서 다양한 실험에 의해 검증이 되어진 모습을 보고 채택하게 되었다.
        - 4:1을 만족하는 128*512size에서 훨씬 좋은 성능을 보이는 상황이였기에 128*512=256*256를 감안하여 256*256으로 논문실험이 진행되어진 152, 200을 사용하게 되었다.
        - 270, 350 또한 256*256으로 논문에서는 실험이 되었지만 많은 Parameter를 가져 GPU Out Of Memory Error를 보였고, Batchsize를 줄여 이를 해결하게 되더라도 훨씬 오랜 학습시간을 가져야 했기에 실험군에서 제외했다.
        - ResnetRS152, ResnetRS200을 실제로 학습시에 큰 성능향상이 있었고, ResnetRS200은 152에 비해 더 많은 학습시간을 가졌지만 성능은 크게 차이가 없었기에 이후 실험들은 ResnetRS152를 CNN Block으로 가지고 진행되었다.

            ![Project%20Detail/Untitled%2014.png](Project%20Detail/Untitled%2014.png)

    5. NFnet
        - 현재 Imagenet에서 높은 성능을 보이고있는 NFnet에 대한 실험 또한 이루어졌다.

            ![Project%20Detail/Untitled%2015.png](Project%20Detail/Untitled%2015.png)

        - 학습 초기에는 큰 성능을 보이다가 중반부를 다가설때 쯔음 성능이 감소하는 현상을 보였다. Accuracy만 감소하는 것이 아니라 Loss또한 같이 증가하는 현상을 보였다.
        - 왜 그럴까?
            - NFnet의 논문에서는 Normalization을 없애는 대신에 다양한 요소들을 통해 이를 보완했다. 그렇지만 현재 많은 실험들이 진행되어진 상태에서 이러한 요소들을 적용하기에는 이미 실험되어진 요소들을 다시 검증하는 과정을 거쳐야 하기에 무리가 있었다.
            - Parameter의 관점에서 ResnetRS 152와 비교를 해보았다.

                | Model | Params Num |
                | -------- | -------- |
                | ResnetRS 152     | 84,572,576     |
                | ResnetRS 200 | 91,160,992 |
                | NFnet F0     | 68,371,360    |
                | NFnet F1 | 129,478,560 |
                | NFnet F2    | 190,585,760     |



            - 위 표의 내용을 통해 현재 구성되어진 Transformer 구조에서는 80,000,000~100,000,000 Params를 가진 CNN Block이랑 합이 맞는 모습을 보인다고 판단이 되었다.
            - 따라서 NFnet또한 실험군에서 제외되었다.

- Encoder
    - QKV sharing : SATRN 학습시간이 오래 걸려 학습 속도 향상을 위해 baseline 확인 중 Query, Key, Value EncoderLayer 부분에서 각각 input으로 들어가고 있어 이를 qkv로 묶어주는 작업을 진행했습니다. 성능, 속도 면에서 미세하게 향상이 있었습니다.

        ![Project%20Detail/Untitled%2016.png](Project%20Detail/Untitled%2016.png)

    - Residual Attention : CNN에서 사용하던 Residual 방법을 Attention에도 적용할 수 있을 것 같아서 찾아보다 올해 발표된 Google RealFormer이라는 논문에서 Residual Attention을 적용한 것을 확인할 수 있었습니다. 이를 저희 Task에도 적용해보면 좋을 것 같았고, EncoderLayer에서 이전의 정보를 추가적으로 넘겨주는 Residual Attention 부분을 추가해서 모델을 개선하였습니다.
         <p align="center">
         <img height="400" src="Project%20Detail/Untitled%2017.png">
         </p>

    - Positional Encoder 수정
        <p align="center">
        <img height="400" src="Project%20Detail/_2021-06-13__9.06.22.png">
        </p>

        Adaptive 2D Positional Encoding은 이미지의 텍스트가 수직, 수평으로 정렬되어있을 때 이미지에서 어느 방향의 정보가 더 중요한지 ConvNet을 통해 스스로 학습하여 수직, 수평 방향의  중요도를 결정하게 됩니다. 

        이미지 feature map에 convolution을 적용한 뒤 average pooling을 적용하고 positional encoding과 함께 더해주는 방식으로 진행하였습니다.

    - Rezero 적용

        [rezero란](https://arxiv.org/abs/2003.04887)?

        residual 연산시 학습 가능한 파라미터 $\alpha$를 통해 특정 레이어에서 중요하지 않은 파라미터들의 가중치를 조절함으로써 학습을 원활하게 동작하게 하는 방법론입니다.
        <p align="center">
        <img height="400" src="Project%20Detail/_2021-06-14__1.01.51.png">
        </p>

        Transformer의 Encoder와 Decoder의 Feed forward 함수에서 LayerNorm을 삭제한 뒤 학습 가능한 파라미터 $\alpha$를 곱해주는 방식으로 진행하였습니다.

        ![Project%20Detail/_2021-06-14__1.11.32.png](Project%20Detail/_2021-06-14__1.11.32.png)

    - Feed forward 수정

        <p align="center">
        <img height="400" src="Project%20Detail/_2021-06-13__5.18.27.png">
        </p>

        - Transformer encoder의 feedforward 레이어를 convolution으로 변경하여 이미지 세로 한줄이 아닌 문자 주변의 locality를 보며 학습을 하도록 진행하였습니다.
        - (b) Convolution 파라미터 갯수: 약 2억개
        (c) Separable depthwise convolution: 약 1억개
        - 논문에선 (b)Convolution을 적용하였을 때 가장 성능이 높았다고 제시하여 두 방법을 모두 활용해서 비교했을 때, (c)Separable보다 (b)Convolution에서 더 좋은 성능을 얻었습니다. 최종적으로 (b)Convolution를 encoder의 feedforward에 이용하였습니다.
        (c)Separable이 파라미터 개수가 적어 GPU메모리 상 이득을 얻을 수 있었습니다. 추후 GPU 메모리 적은 환경에서는 비교적 좋은 성능을 얻을 수 있을 것이라고 판단됩니다.
- Decoder
    - 처음 Baseline으로 주어진 model을 순서대로 이해하고 변경을 진행하다 보니 늦게 Decoder part를 보게 되었다.
    - decoder layer하나에 attention이 2번 진행되어지도록 구성되어져있고, first attention의 out이 다음 second attention의 input으로 들어가도록 구성이 되어지는것이 일반적이지만, baseline에서 실수로인해 first attention의 input과 second attention의 input이 같은 input이 적용하도록 되어 2 branch구조를 띄게 구성되어있었다.
    - 이를 수정하여 원 의도대로 이어진 구조로 만들어 실험을 진행하였다.

        ![Project%20Detail/Untitled%2018.png](Project%20Detail/Untitled%2018.png)

    - 초중반에 학습이 차이가 났지만, 수렴지점은 비슷한 모습을 보였다.
    - 구조가 다르더라도, 현재 model에서 decoder 가 상대적으로 다른 layer들에 비해 얕게 구성되어져있으며 수정 전,후가 비슷한 맥락으로 forward가 진행되어지는 구조이기에 큰 변화는 없었던 것 같다.

- Activation Function
    1. RELU(Default)
        - Transformer의 FeedForward의 활성화 함수가 RELU로 구성 되어있다.
        - Transformer는 q,k,v layer가 전부 Linear로 구성되어있으며 FeedForward또한 Linear로 이루어지고있어, 이를 RELU를 통해 Activation을 진행하는 것 보다 0이하의 값들을 어느정도 살리는 것이 어떨까? 하는 생각으로 아래의 두실험이 진행되었다.
    2. Leaky RELU
        - ASTER로 실험이 진행되었다.

            ![Project%20Detail/Untitled%2019.png](Project%20Detail/Untitled%2019.png)

        - LeakyRELU를 적용했을때에 더 안좋은 성능을 보였다.
            - 왜그럴까?
                - baseline이 ASTER, SATRN이라는 이름으로 주어졌지만 세부적으로 보면 Customizing이 많이 포함되어진 모습을 볼수있었고, 이러한 과정들을 통해 RELU에 최적화가 이미 되어진 구조라고 판단이 되었다.
                - 조금의 차이에 비해 위의 결과를 보면 큰 차이가 발생하는 모습도 위의판단의 근거가 되었다.
    3. GELU
        - 조금 더 섬세한 비교를 위해 SATRN으로 진행되었다.

            ![Project%20Detail/Untitled%2020.png](Project%20Detail/Untitled%2020.png)

        - 위의 Leaky RELU의 경우처럼 성능이 좋지않았다.
        - 이에 대한 이유 또한 위의 Leaky RELU와 비슷하다고 판단이 되었다.

### ViT

- Backbone
    - ResNetV2
        - 

- Encoder
    - VisionTransformer

- Decoder
    - TransformerWrapper

- 결과 및 분석
    - Train Score, Validation Score 차이가 너무 큼 → Augmentation이 적합하지 않다는 결론(실제 데이터와 다른 방향의 증강을 한 것 같음) → Augmentation 제거
    - 문맥은 맞으나 특정 Symbol을 예측하지 못하는 경우가 많음. ex) 1과 i를 헷갈림 등 → Encoder, Backbone model 이 Deep하지 않아서 그런게 아닐까? → ViT patch 감소(16→4)
    - Loss가 너무 낮음 → Pad token을 loss 계산 시 ignore해주지 않아 전체적으로 loss값이 작아진듯 → ignore_index 추가
    - 학습 중 score가 더디게 항상 증가하는 경향 → lr이 너무 낮은 것 같음 + pad ignore → lr 증가

### Swin Transformer

## Loss

### CrossEnropy Loss

- CrossEnropy Loss
    - Ignore index = False vs True
        - 초기에는 월등히 좋은 성능을 보이다가 어느 순간 성능이 급격하게 떨어졌다

            ![Project%20Detail/Untitled%2021.png](Project%20Detail/Untitled%2021.png)

        - Log를 통해 결과를 확인해보니, ignore_index에 PAD Token이 적용되기에 PAD자리가 단순히 EOS로 대체가 되어지거나, 예측 중간에 PAD가 나올경우 계산이 끊기는 현상이 발생했다.
        - 이를 수정하기에 너무많은 변수들이 발생할 것 같은 위험과 다른 여러 실험들에 더 흥미가 있었기에 Ignore_index는 사용하지 않는 방향으로 실험이 진행되었다.

### Label Smoothing CrossEntropy Loss

- Label Smoothing CrossEntropy Loss
    - 학습의 일반화를 높여주기 위해 Label Smoothing을 적용하면 어떨까?

         → 적용전에 비해 낮은 성능

        ![Project%20Detail/Untitled%2022.png](Project%20Detail/Untitled%2022.png)

    - Train Data가 8만개의 적지않은 양과, Augmentation이 natural하게 들어있고 Layer norm, Batch Norm또한 많이 들어있어 이미 다양한 Generalize가 적용되어있었기에 Label Smoothing을 적용하게되면 오히려 성능이 그대로 감소하게 되는 효과를 보인 것 같다.

### Cosine Loss, Focal Loss, Cosine + Focal Loss

- Cosine Loss, Focal Loss, Cosine + Focal Loss
    - SATRN 학습시간이 45시간, ASTER 학습시간이 5시간이기에 Focal Loss, Cosine Loss 실험은 ASTER로 진행했다.
    - Kaggle의 여러 Discussion을 읽어보던 중, Cosine+Focal Loss관련 내용이 있었다.([https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271))

        ![Project%20Detail/Untitled%2023.png](Project%20Detail/Untitled%2023.png)

    - CosineEmbeddingLoss를 통해 Pred, Target Sentence전체에 대한 Loss계산이 이루어지기에 token마다 분류해주는 CrossEntropy Loss보다 조금더 Sentence에 초점을 맞출수 있을 것 같다는 판단으로 더 좋은 성능이 나오지 않을까 하는 기대와 함께 실험을 진행했다.
    - Weight 비율을 여러 방향으로 조절하며 다양한 방향으로 실험을 진행했지만 낮은 성능이 나왔다.
    - 왜 그럴까?
        - Cosine Loss를 계산할때에 PAD, SOS, END와 같은 Special Token들도 같이 유사도 계산이 진행되어 위와 같은 결과가 발생했다.
        - 이를 해결하기 위해서는 Special Token에 대한 예외 처리를 진행해 주어야 하는 Ignore_index의 모습과 똑같았기에 더 이상의 실험은 진행하지 않았다.

### Ensemble

- 기존의 inference 방식에 모델 checkpoint를 불러오는 방식을 적용했을 때, 서버 GPU가 Out of Memory 에러가 발생합니다.  문제를 해결하기 위해 2가지 방법을 생각했습니다.
    - inference를 할 때, 모델을 순차적으로 불러오면서 데이터 전체를 예측한 것을 저장해두는 방식 → 이미지가 계속 gpu에 쌓여서 문제가 발생
    - batch 마다 모델을 변경하면서 예측하도록 구성하니 메모리가 터지는 현상은 비교적 줄었으나, batch size 8일 때는 여전히 Memory 문제 발생 → batch size 4로 줄여서 해보니 비교적 안정적으로 예측 가능했습니다.

## Future Work


- Albumentations를 통한 augmentation → 라이브러리 충돌로 인해서 사용 불가
- Swin Transformer에 대해 섬세한 실험.
- 다양한 Pretrained Model에 대한 실험.
- Ignore Index의 문제점 개선.
- Image rectification을 포함한 End-to-End 모델 개발.
- CSTR, SEED와 같은 다른 structure 모델 구축.
- 순차적으로 구조를 변경했었기에 Decoder부분에 대해 다양한 실험을 진행하지 못함.
- Beam Search 구현.
- Multi Head Attention의 collapse문제를 해결하기 위해  Bayesian 논리를 적용해서 해결한 논문 구현.
([https://arxiv.org/pdf/2009.09364.pdf](https://arxiv.org/pdf/2009.09364.pdf))
- ASTER에 대한 추가적인 개발.
- Guided Training에 대한 추가적인 개발. (ex. ASTER + SATRN, SATRN + Discriminator, etc...)
- ASTER에서 LSTM을 BERT Layer로 교체.
- GAN을 이용한 Datacleaning 개발.
- Level, Source 등 추가적인 정보들을 활용하여 모델 개발.
- 기울어진 이미지 올바르게 적용하는 skew 실패.
- 다양한 하이퍼 파라미터 변경.
    - Dropout, lr scheduler, teacher forcing scheduler 변경

# Reference


- RealFormer: Transformer Likes Residual Attention ([https://arxiv.org/pdf/2012.11747.pdf](https://arxiv.org/pdf/2012.11747.pdf))
- On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention ([https://arxiv.org/pdf/1910.04396v1.pdf](https://arxiv.org/pdf/1910.04396v1.pdf))
- ReZero is All You Need: Fast Convergence at Large Depth ([https://arxiv.org/abs/2003.04887](https://arxiv.org/abs/2003.04887))
- ASTER: An Attentional Scene Text Recognizer with Flexible Rectification ([http://122.205.5.5:8071/UpLoadFiles/Papers/ASTER_PAMI18.pdf](http://122.205.5.5:8071/UpLoadFiles/Papers/ASTER_PAMI18.pdf))
- EfficientNetV2: Smaller Models and Faster Training ([https://arxiv.org/pdf/2104.00298](https://arxiv.org/pdf/2104.00298))
- Repulsive Attention:Rethinking Multi-head Attention as Bayesian Inference ([https://arxiv.org/pdf/2009.09364.pdf](https://arxiv.org/pdf/2009.09364.pdf))
- Revisiting ResNets: Improved Training and Scaling Strategies ([https://arxiv.org/abs/2103.07579](https://arxiv.org/abs/2103.07579))
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks ([https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593))
- Contrastive Learning for Unpaired Image-to-Image Translation ([https://arxiv.org/pdf/2007.15651.pdf](https://arxiv.org/pdf/2007.15651.pdf))
- High-Performance Large-Scale Image Recognition Without Normalization ([https://arxiv.org/abs/2102.06171](https://arxiv.org/abs/2102.06171))
- Attention Is All You Need ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows ([https://arxiv.org/pdf/2103.14030v1.pdf](https://arxiv.org/pdf/2103.14030v1.pdf))
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ([https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929))
- GTC: Guided Training of CTC Towards Efficient and Accurate Scene Text Recognition
([https://arxiv.org/pdf/2002.01276.pdf](https://arxiv.org/pdf/2002.01276.pdf))
