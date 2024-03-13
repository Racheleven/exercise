import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.fc1 = nn.Linear(1, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def myfunction(x):
    y=4*(x+1)**3+x**2
    return y

# 采集数据
X = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
Y = myfunction(X)
# 打乱数据
indices = torch.randperm(X.size(0))
X = X[indices]
Y = Y[indices]
# 划分训练集和测试集
trainX = X[:600]
trainY = Y[:600] + torch.normal(0, 0.1, size=(600, 1))
testX = X[600:]
testY = Y[600:]

# 模型
model = myNet()
best_loss = float('inf')  # 初始化最佳训练集损失为正无穷大
best_model = None
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 5000
train_loss=[]
for epoch in range(epochs):
    outputs = model(trainX)
    loss = criterion(outputs, trainY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())  # append the loss for this epoch

    if loss < best_loss:
        best_loss = loss
        best_model = model.state_dict()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    model.load_state_dict(best_model)
    testY_pred= model(testX)
    test_loss = criterion(testY_pred, testY)
    print(f'测试集上的loss: {test_loss.item():.4f}')


# 绘制预测点和真实点
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.scatter(trainX.numpy(), trainY.numpy(), color='blue', label='True data')
plt.scatter(trainX.numpy(), trainY.numpy(), color='red', label='Predicted data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Train Dataset')
plt.legend()
plt.show()

# 绘制预测点和真实点
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.scatter(testX.numpy(), testY.numpy(), color='blue', label='True data')
plt.scatter(testX.numpy(), testY_pred.numpy(), color='red', label='Predicted data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Dataset')
plt.legend()
plt.show()