from imports import *
class MultiHeadAttention(nn.Module):
    def __init__(self, embeddingDim=768, numHeads=12, attentionDropout=0.0):
        super().__init__()
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.attentionDropout = attentionDropout

        # Key, Value, Query projections
        self.key = nn.Linear(self.embeddingDim, self.embeddingDim)
        self.value = nn.Linear(self.embeddingDim, self.embeddingDim)
        self.query = nn.Linear(self.embeddingDim, self.embeddingDim)
        self.output = nn.Linear(self.embeddingDim, self.embeddingDim)

    def forward(self, hidden_states):
        batchSize, numPatches, embeddingSize = hidden_states.shape
        
        
        # Query, Key, Value
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape for multi-head attention (batch-size, numberofpatches, numberofheads, embbeding/numheads)
        query = query.view(batchSize, numPatches, self.numHeads, embeddingSize // self.numHeads).transpose(1, 2)
        key = key.view(batchSize, numPatches, self.numHeads, embeddingSize // self.numHeads).transpose(1, 2)
        value = value.view(batchSize, numPatches, self.numHeads, embeddingSize // self.numHeads).transpose(1, 2)

        # Compute Attention weights
        attn_weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attn_weights = F.softmax(attn_weights, dim=-1).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attentionDropout, training=self.training)

        # Apply attention weights to values
        attentionOutput = attn_weights @ value

        # Reshape back to (batchSize, numPatches, embeddingSize)
        attentionOutput = attentionOutput.transpose(1, 2).reshape(batchSize, numPatches, embeddingSize).contiguous()
        attentionOutput = self.output(attentionOutput)

        return attentionOutput

# Example usage
batchSize = 1
numPatches = 196
embeddingDim = 768

# Check if input size matches output size
hidden_states = torch.randn(batchSize, numPatches, embeddingDim)
attention = MultiHeadAttention(embeddingDim=embeddingDim, numHeads=12, attentionDropout=0.0)
output = attention(hidden_states)

print(f"Input shape: {hidden_states.shape}")
print(f"Output shape: {output.shape}")