from attentionModule import *
from embeddingModule import *
from encodingModule import *
from MLP import *

class VisionTransformer(nn.Module):
    def __init__(self, embeddingFunction, encoderFunction, embeddingDim, layer_norm_eps):
        super().__init__()
        self.embeddingFunction = embeddingFunction
        self.encoderFunction = encoderFunction
        self.post_layernorm = nn.LayerNorm(embeddingDim, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.embeddingFunction(hidden_states)
        embeddingOutput = self.encoderFunction(hidden_states)
        encodedOutput = self.post_layernorm(embeddingOutput)
        return encodedOutput


class FullVisionTransformer(nn.Module):
    def __init__(self, embeddingFunction, encoderFunction, embeddingDim, layer_norm_eps):
        super().__init__()
        self.vision_model = VisionTransformer(embeddingFunction, encoderFunction, embeddingDim, layer_norm_eps)

    def forward(self, image_tensor):
        return self.vision_model(image_tensor)



batchSize = 1
numPatches = 196
embeddingDim = 768
numChannels = 3
imageSize=224
layer_norm_eps = 1e-6
patchSize = 16

# Generate Random values
random_image_tensor = torch.randn(batchSize, numChannels, imageSize, imageSize)

# Custom Embedding Output
embeddingFunction = Embeddings(numChannels=numChannels, embeddingDim=embeddingDim, imageSize=imageSize, patchSize=patchSize)
encoderFunction = EncoderModule(numEncoders=12, firstLayerSize=embeddingDim, numHeads = 12, secondLayerSize=3072)
visionTransformer = FullVisionTransformer(embeddingFunction = embeddingFunction, encoderFunction = encoderFunction,embeddingDim = embeddingDim, layer_norm_eps = layer_norm_eps)

visionTransformer(random_image_tensor).shape



