import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from pytocl.preprocess import create_torch_dataset


# Hyper Parameters
input_size = 21
hidden_size = 40
output_size = 3
num_epochs = 50
learning_rate = 0.001

x_train, y_train, x_test, y_test = create_torch_dataset()

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearRegression, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        # self.i2h = sigmoid(self.i2h)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # out = self.linear(x)
        hidden = self.sigmoid(self.i2h(x))
        out = self.h2o(hidden)
        return out


model = LinearRegression(input_size, output_size, hidden_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print ('Epoch [%d/%d], Loss: %.4f'
               %(epoch+1, num_epochs, loss.data[0]))

# Save the Model
torch.save(model.state_dict(), 'model_nn_regression.pkl')
