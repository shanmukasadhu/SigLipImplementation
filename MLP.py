from imports import *
class MLP(nn.Module):
    # 2 Dense/Fully-Connected Layers
    def __init__(self, firstLayerSize, secondLayerSize):
        super().__init__()
        # Define the first Dense Layer: size 768 -> size 3072
        self.fc1 = nn.Linear(firstLayerSize, secondLayerSize)
        # Define the first Dense Layer: size 3072 -> size 768
        self.fc2 = nn.Linear(secondLayerSize, firstLayerSize)

    def forward(self, hidden_states: torch.Tensor):
        # First Layer
        layer1 = self.fc1(hidden_states)
        # TanH Gelu Activation
        activationFunction1 = F.gelu(layer1, approximate="tanh")
        # Second Layer
        layer2 = self.fc2(activationFunction1)
        return layer2

batchSize = 1
numPatches = 196
embeddingDim = 768

# Check if input size matches output size
hidden_states = torch.randn(batchSize, numPatches, embeddingDim)



# Example
mlp = MLP(firstLayerSize=768, secondLayerSize=3072)
output = mlp(hidden_states)
print(output.shape)  
