#신경망(Neural Networks)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 5x5의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5은 이미지 차원에 해당
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # 합성곱(Convolution) 레이어 c1: 입력 이미지 채널 1, 출력 채널 6,
        # 5x5 정사각 합성곱, 활성 함수로 RELU 사용 및 (N, 6, 28, 28)의 크기를
        # 갖는 Tensor를 출력 (N은 배치 크기)
        c1 = F.relu(self.conv1(input))
        # 서브샘플링(Subsampling) 레이어 s2: 2x2 격자, 순전히 기능적인 레이어로,
        # 이 레이어는 어떠한 매개변수도 가지지 않고 (N, 6, 14, 14) Tensor를 출력
        s2 = F.max_pool2d(c1, (2, 2))
        # 합성곱(Convolution) 레이어 c3: 입력 채널 6, 출력 채널 16,
        # 5x5 정사각 합성곱, 활성 함수로 RELU 사용 및 (N, 16, 10, 10)의 크기를
        # 갖는 Tensor를 출력
        c3 = F.relu(self.conv2(s2))
        # 서브샘플링(Subsampling) 레이어 s4: 2x2 격자, 순전히 기능적인 레이어로,
        # 이 레이어는 어떠한 매개변수도 가지지 않고 (N, 16, 5, 5) Tensor를 출력
        s4 = F.max_pool2d(c3, 2)
        # 평탄화(flatten) 연산: 순전히 기능적으로 동작하며, (N, 400) Tensor를 출력
        s4 = torch.flatten(s4, 1)
        # 완전히 연결된 레이어 f5: (N, 400) Tensor를 입력으로 받아서
        # (N, 120) Tensor를 출력하며, 활성 함수로 RELU 사용
        f5 = F.relu(self.fc1(s4))
        # 완전히 연결된 레이어 f6: (N, 120) Tensor를 입력으로 받아서
        # (N, 84) Tensor를 출력하며, 활성 함수로 RELU 사용
        f6 = F.relu(self.fc2(f5))
        # 가우시안 레이어 출력: (N, 84) Tensor를 입력으로 받아서
        # (N, 10) Tensor를 출력
        output = self.fc3(f6)
        return output

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면, 하나의 숫자만을 특정(specify)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # 배치 차원을 제외한 모든 차원을 하나로 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1의 .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
""" input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss """

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듦

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 새로운 가중치 = 가중치 - 학습률 * 변화도
#weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)은 다음과 같습니다:
optimizer.zero_grad()   # 변화도 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트 진행