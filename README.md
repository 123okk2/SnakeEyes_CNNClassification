# SnakeEyes_CNNClassification
모델별 SnakeEyes (주사위) 눈금 인식 정확도 비교

## 주제
CNN의 대표적인 다섯 가지 모델을 활용, 100만개의 데이터 셋으로 학습을 시킨 후 1만개의 테스트 셋으로 정확도를 판별한다.

## 모델 목록
1. LeNet-5 모델
2. AlexNet 모델
3. VGGNet 모델
4. ResNet 모델
5. GoogleNet 모델

## SnakeEyes
데이터 자체는 이미 데이터화 되어있다.
### 데이터를 사진으로 출력했을 때의 모습

### Download Data
https://www.kaggle.com/nicw102168/snake-eyes

## 결과

시간은 VGGNet이 가장 오랜 시간을 소모했으나, 대신 고작 5 epoch 만에 100%라는 정확도를 이뤄낼 수 있었다.
LeNet-5는 특징 맵의 개수가 가장 적어 가장 적은 시간을 소모했으나, 그 정확도 또한 고작 92%에 그치고 말았다.
시간 대비 가장 좋은 효율을 보인 것은 당연 GoogleNet이었다. LeNet-5 외에 가장 적은 시간을 소모했으며, 10 epoch만에 99.78%이라는, VGGNet을 제외하고 가장 높은 정확도를 보여줬다.
