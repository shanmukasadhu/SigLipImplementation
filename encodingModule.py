# Encoding Layer
from attentionModule import *
from MLP import *

class EncoderLayer(nn.Module):
    def __init__(self, firstLayerSize,numHeads, secondLayerSize, 
                 attentionDropout= 0.0, layer_norm_eps = 1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(firstLayerSize, numHeads, attentionDropout)
        self.normLayer1 = nn.LayerNorm(firstLayerSize, eps=layer_norm_eps)
        self.mlp = MLP(firstLayerSize, secondLayerSize)
        self.normLayer2 = nn.LayerNorm(firstLayerSize, eps=layer_norm_eps)

    def forward(self, hidden_states):
        # Implements the residual connections of the Encoder Layer
        residual = hidden_states
        normalizedResidual1 = self.normLayer1(hidden_states)
        attentionLayer = self.attention(normalizedResidual1)
        firstStep = residual + attentionLayer

        
        normalizedResidual2 = self.normLayer2(firstStep)
        mlpLayer = self.mlp(normalizedResidual2)
        output = residual + mlpLayer
        return output


batchSize = 1
numPatches = 196
embeddingDim = 768


# Check if input size matches output size
hidden_states = torch.randn(batchSize, numPatches, embeddingDim)

encoder_layer = EncoderLayer(firstLayerSize=768, numHeads = 12, secondLayerSize=3072)
encoder_layer(hidden_states).shape

# Encoding Module


class EncoderModule(nn.Module):
    def __init__(self, numEncoders, firstLayerSize,numHeads, secondLayerSize, 
                 attentionDropout= 0.0, layer_norm_eps = 1e-5):
        super().__init__()
        self.encoderLayers = nn.ModuleList()
        for i in range(numEncoders):
            layer = EncoderLayer(firstLayerSize, numHeads, secondLayerSize)
            self.encoderLayers.append(layer)



    def forward(self, hidden_states):
        for layer in self.encoderLayers:
            hidden_states = layer(hidden_states)
        return hidden_states

batchSize = 1
numPatches = 196
embeddingDim = 768

# Check if input size matches output size
hidden_states = torch.randn(batchSize, numPatches, embeddingDim)


encoder = EncoderModule(numEncoders=12, firstLayerSize=768, numHeads = 12, secondLayerSize=3072)
print(encoder(hidden_states).shape)



