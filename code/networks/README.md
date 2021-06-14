## Model 설명
### SATRN_3.py
* CNN : ResnetRS152
* custom initialize
* Positional-Encoding 수정
* Encoder : Residual-Attention 추가, Feed-forward 수정(convolution)
* Decoder : original attention
* Dataset : train(80K)

### SATRN_4.py
* SATRN_3.py와 동일
* Decoder : 2 branch attention
* Dataset : train(80K)

### SATRN_adamP.py
* SATRN_4.py와 동일
* Optimizer : Adam -> AdamP
* Dataset : train all(100K)

### SATRN_extension.py
* SATRN_4.py와 동일
* DecoderLayer number : 3 -> 6
* Dataset : train all(100K)

### SATRN_Final_all.py
* SATRN_4.py와 동일
* Dataset : train all(100K)

### EFFICIENT_SATRNv6.py
* 기본적인 SATRN 구조는 SATRN_4.py와 동일 
* CNN : Efficientnetv2 small
* Encoder Filter dimension : 512 -> 1024
* lr : 5e-4 -> 4e-4
* dropout rate : 0.1 -> 0.3
* Dataset : train all(100K)
