import numpy as np
import tflearn
import matplotlib.pyplot as plt
import random

#------1. 데이터 셋 불러오기
#test 셋 : 1만 개의 데이터
test = np.vstack(tuple(np.fromfile("snake-eyes/snakeeyes_test.dat", dtype=np.uint8).reshape(-1,401)))
x_test = test[:, 1:]
y2 = test[:, 0]

#class 갯수인 12로 맞추기 위한 (reshape 하기 위한) 함수
y_test = np.zeros([len(y2), 12], dtype=bool)
for i in range(len(y2)):
    label = y2[i]
    y_test[i, label - 1] = True

#------2. 표준 스케일링
mean = x_test.mean(axis=0)
std=x_test.std(axis=0)

x_test = (x_test - mean)/std
x_test = x_test.reshape(-1, 20, 20, 1)
'''
#1. LeNet-5
network = tflearn.input_data(shape=[None, 20, 20, 1], name='input')
network = tflearn.conv_2d(network, 10, 3, activation='tanh', padding='same') #20-3+1=18
network = tflearn.avg_pool_2d(network, 2, strides=2, padding='valid') #18/2=9
network = tflearn.conv_2d(network, 15, 2, activation='tanh', padding='valid') #9-2+1=8
network = tflearn.avg_pool_2d(network, 2, strides=2, padding='valid') #8/2=4
network = tflearn.conv_2d(network, 18, 1, activation='tanh', padding='valid') #4-1+1=4
network = tflearn.avg_pool_2d(network, 2, strides=2, padding='valid') #4/2=2
network = tflearn.conv_2d(network, 21, 2, activation='tanh', padding='valid') #2-2+1=1
network = tflearn.fully_connected(network, 84, activation='tanh')
network = tflearn.fully_connected(network, 12, activation='softmax')
network = tflearn.regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

model=tflearn.DNN(network, tensorboard_verbose=0)

model.load(model_file='TestModels/LeNet5Model')
'''
'''
network = tflearn.input_data(shape=[None, 20, 20, 1])
network = tflearn.conv_2d(network, 96, 3, strides=1, activation='relu', padding='valid') #20-3+1=18
network = tflearn.max_pool_2d(network, 2, strides=2, padding='valid') #18/2=9
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 256, 3, activation='relu', padding='same') #18-3+1=16
network = tflearn.max_pool_2d(network, 2, strides=2, padding='valid') #16/2=8
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 384, 3, activation='relu', padding='same') #8-3+1=6
network = tflearn.conv_2d(network, 384, 3, activation='relu', padding='same') #6-3+1=4
network = tflearn.conv_2d(network, 256, 3, activation='relu', padding='same') #4-3+1=2
network = tflearn.max_pool_2d(network, 2, strides=2, padding='valid') #2/2=1
network = tflearn.local_response_normalization(network)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 12, activation='softmax')
network = tflearn.regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=2)
model.load(model_file='TestModels/AlexNetModel')
'''
'''
#VGGNet 실험
input = tflearn.input_data(shape=[None, 20, 20, 1]) #20
c1= tflearn.conv_2d(input, 96, 3, activation='relu') #
c2= tflearn.conv_2d(c1, 256, 3, activation='relu') #
m1 = tflearn.max_pool_2d(c2, 2, strides=2) #
c3 = tflearn.conv_2d(m1, 128, 3, activation='relu')
c4 = tflearn.conv_2d(c3, 128, 3, activation='relu')
m2 = tflearn.max_pool_2d(c4, 2, strides=2)
c5 = tflearn.conv_2d(m2, 256, 3, activation='relu')
c6 = tflearn.conv_2d(c5, 256, 3, activation='relu')
c7 = tflearn.conv_2d(c6, 256, 3, activation='relu')
m3 = tflearn.max_pool_2d(c7, 2, strides=2)
c8 = tflearn.conv_2d(m3, 512, 3, activation='relu')
c9 = tflearn.conv_2d(c8, 512, 3, activation='relu')
c10 = tflearn.conv_2d(c9, 512, 3, activation='relu')
m4 = tflearn.max_pool_2d(c10, 2, strides=2)
c11 = tflearn.conv_2d(m4, 512, 3, activation='relu')
c12 = tflearn.conv_2d(c11, 512, 3, activation='relu')
c13 = tflearn.conv_2d(c12, 512, 3, activation='relu')
m5 = tflearn.max_pool_2d(c13, 2, strides=2)
f1 = tflearn.fully_connected(m5, 4096, activation='relu')
d1 = tflearn.dropout(f1, 0.5)
f2 = tflearn.fully_connected(d1, 4096, activation='relu')
d2 = tflearn.dropout(f2, 0.5)
f3 = tflearn.fully_connected(d2, 12, activation='softmax')

network = tflearn.regression(f3, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load("TestModels/VGGNetModel")
'''
'''
#ResNet 실험
net = tflearn.input_data(shape=[None, 20, 20, 1])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False) #20-3+1 = 18
# 잔차 블록
net = tflearn.residual_bottleneck(net, 3, 16, 64) #18-2 = 16
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.residual_bottleneck(net, 1, 128, 512, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 128, 512)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(net, 12, activation='softmax')
net = tflearn.regression(net, optimizer='momentum',
	loss='categorical_crossentropy', learning_rate=0.1)
# 학습
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load("TestModels/ReNet5")
'''
#GoogleNet
def inception(input, n1, n2, n3, n4, n5, n6) :
	reduce1 = tflearn.conv_2d(input, n1, 1, activation='relu')
	reduce2 = tflearn.conv_2d(input, n2, filter_size=1, activation='relu')
	pool = tflearn.max_pool_2d(input, kernel_size=3, strides=1)
	conv1 = tflearn.conv_2d(input, n3, 1, activation='relu')
	conv2 = tflearn.conv_2d(reduce1, n4, filter_size=3, activation='relu')
	conv3 = tflearn.conv_2d(reduce2, n5, filter_size=5, activation='relu')
	conv4 = tflearn.conv_2d(pool, n6, filter_size=1, activation='relu')
	output = tflearn.merge([conv1, conv2, conv3, conv4], mode='concat', axis=3)
	return output

input = tflearn.input_data(shape=[None, 20, 20, 1])
conv1 = tflearn.conv_2d(input, 64, 7, strides=2, activation='relu') #20/2=10 #20-6=14
m1 = tflearn.max_pool_2d(conv1, 3, strides=2) #14/2=7
l1 = tflearn.local_response_normalization(m1)
conv2 = tflearn.conv_2d(l1, 64, 1, activation='relu') #14
conv3 = tflearn.conv_2d(conv2, 192, 3, activation='relu') #12
l2 = tflearn.local_response_normalization(conv3)
m2 = tflearn.max_pool_2d(l2, kernel_size=3, strides=2) #6
i1 = inception(m2, 96, 16, 64, 128, 32, 32)
i2 = inception(i1, 128, 32, 128, 192, 96, 64)
m3 = tflearn.max_pool_2d(i2, kernel_size=3, strides=1) #3
i3 = inception(m3, 96, 16, 192, 208, 48, 64)
i4 = inception(i3, 112, 24, 160, 224, 64, 64)
i5 = inception(i4, 128, 24, 128, 256, 64, 64)
i6 = inception(i5, 144, 32, 112, 288, 64, 64)
i7 = inception(i6, 160, 32, 256, 320, 128, 128)
m4 = tflearn.max_pool_2d(i7, kernel_size=3, strides=1) #
i8 = inception(m4, 160, 32, 256, 320, 128, 128)
i9 = inception(i8, 192, 48, 384, 384, 128, 128)
a1 = tflearn.avg_pool_2d(i9, kernel_size=7, strides=1)
d1 = tflearn.dropout(a1, 0.4)
f1 = tflearn.fully_connected(d1, 12, activation='softmax')
network = tflearn.regression(f1, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)
# 학습
model = tflearn.DNN(network, tensorboard_verbose=2)
model.load('TestModels/GoogleNet')

print("Accuracy : %.4f" %model.evaluate(x_test, y_test)[0])
n2=random.sample(range(0,10000),20)
for k in range(20):
    plt.subplot(2, 10, k+1)
    
    #결과값으로 제목 설정
    num=model.predict(x_test[n2[k]].reshape(-1,20,20,1))
    nums=model.predict(x_test[n2[k]].reshape(-1,20,20,1))
    nums.sort()
    plt.title(num[0].tolist().index(nums[0][len(nums[0])-1])+1)

    #이미지 출력
    plt.imshow(x_test[n2[k]].reshape(20,20))
    plt.axis('off')
    plt.axis('off')

plt.show()