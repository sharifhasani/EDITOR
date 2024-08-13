from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch


PATH = "./vitb_16_224_21k.pth"
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
torch.save(model, PATH)