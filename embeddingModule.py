from imports import *
class Embeddings(nn.Module):
    def __init__(self, numChannels=3, embeddingDim=768, imageSize=224, patchSize=16):
        super().__init__()
        
        # Number of channels will be 3 since it's an RGB Image 
        self.numChannels = numChannels
        # Dimension of Patch Embedding numChannels * pathSize * patchSize: (3x16x16) = 768
        self.embeddingDim = embeddingDim
        # Image Size = 224
        self.imageSize = imageSize
        # Patch Size = 16
        self.patchSize = patchSize
        
        # Create Patch Embeddings using Convolution Layers
        self.patch_embedding = nn.Conv2d(in_channels=self.numChannels,out_channels=self.embeddingDim,kernel_size=self.patchSize,stride=self.patchSize,padding="valid",)
        
        # Calculate number of patches
        self.numPatches = (self.imageSize // self.patchSize) ** 2
        # Calculate position embeddings using PyTorch Embedding Module
        self.positionEmbedding = nn.Embedding(self.numPatches, self.embeddingDim)
        
        self.register_buffer("positionIds",torch.arange(self.numPatches).expand((1, -1)),persistent=False,)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        
        # Calculate Patch Embeddings
        patch_embeds = self.patch_embedding(pixel_values)
        # Reshape Embeddings: Flatten everything 
        embeddings = patch_embeds.flatten(start_dim=2, end_dim=-1)
        embeddings = embeddings.transpose(1, 2)
        # Add Position and Patch Embeddings
        embeddings = embeddings + self.positionEmbedding(self.positionIds)
        return embeddings


# Comparing between SigLip Transformer Implementation

vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")

batchSize = 1
numChannels = 3
imageSize = 224


pixelValues = torch.randn(batchSize, numChannels, imageSize, imageSize)

# Custom Embedding Output
custom_embed_layer = Embeddings()
custom_embeddings = custom_embed_layer(pixelValues)
print(f"Custom Embeddings Shape: {custom_embeddings.shape}")

# Hugging Face Model Embedding Output (Get hidden states)
outputs = vision_model(pixelValues, output_hidden_states=True)
transformer_embeddings = outputs.hidden_states[0]  # First layer hidden state (embeddings)
print(f"Transformers SIGLIP Embeddings Shape: {transformer_embeddings.shape}")