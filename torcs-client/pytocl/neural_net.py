import torch.nn as nn


class FeedForwardRegression(nn.Module):
    """
    Regression Model with sigmoid activation function
    """
    def __init__(self, input_size=21, output_size=3, hidden_size=40):
        super(FeedForwardRegression, self).__init__()

        self.i2h = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.h2o = nn.Linear(hidden_size, output_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        hidden = self.sigmoid(self.i2h(x))
        out = self.h2o(hidden)
        return out
