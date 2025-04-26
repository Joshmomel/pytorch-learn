import torch  # PyTorch deep learning library
import torch.nn as nn  # Neural network module from PyTorch


class TinyNet(nn.Module):
    """
    A simple neural network with a single linear layer
    """
    def __init__(self):
        super(TinyNet, self).__init__()
        # Linear layer that takes 2 input features and outputs 1 value
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        # Forward pass simply applies the linear transformation
        return self.linear(x)
    

# Create an instance of our model
model = TinyNet()

# Create input tensor with 2 features
x = torch.tensor([1.0, 2.0])
# Perform forward pass through the model
y = model(x)
# Print the output
print(y)




        
