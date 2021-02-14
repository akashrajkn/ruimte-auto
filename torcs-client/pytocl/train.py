import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from preprocess import create_torch_dataset
from neural_net import FeedForwardRegression


# Hyper Parameters
input_size = 21
hidden_size = 40
output_size = 3
num_epochs = 50
learning_rate = 0.001

# Train and Test datasets
x_train, y_train, x_test, y_test = create_torch_dataset()

model = FeedForwardRegression(input_size, output_size, hidden_size)

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
realpath = os.path.dirname(os.path.realpath(__file__))
torch.save(model.state_dict(), realpath + '/../../models/neural_net_regression.pkl')
