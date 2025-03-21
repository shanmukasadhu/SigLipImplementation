from imports import *
from attentionModule import *
from embeddingModule import *
from encodingModule import *
from MLP import *
from mainModule import *
from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig

# Obtain SigLip Pre-trained Vision Model from Hugging Face Transformers
vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", config=SiglipVisionConfig(vision_use_head=False))


batchSize = 1
numPatches = 196
embeddingDim = 768
numChannels = 3
imageSize=224
layer_norm_eps = 0.05
image_tensor = torch.randn(batchSize, numChannels, imageSize, imageSize)

# Custom Embedding Output
embeddingFunction = Embeddings(numChannels=3, embeddingDim=768, imageSize=224, patchSize=16)
encoderFunction = EncoderModule(numEncoders=12, firstLayerSize=768, numHeads = 12, secondLayerSize=3072)




mine = FullVisionTransformer(embeddingFunction = embeddingFunction, encoderFunction = encoderFunction,embeddingDim = embeddingDim, layer_norm_eps = layer_norm_eps)



# Get the state dicts
vision_model_state_dict = vision_model.state_dict()  # Vision model's state dict
our_state_dict = mine.state_dict()  # Your model's state dict

renamed_our_state_dict = {}

count = 0
for key, val in vision_model_state_dict.items():
    dict_items = list(our_state_dict.items())

    # Access the nth key-value pair (for example, 2nd item, index 1)
    nth_key, nth_value = dict_items[count]
    count+=1
    renamed_our_state_dict[nth_key]=val

# Should output <All keys matched successfully>
mine.load_state_dict(renamed_our_state_dict)
