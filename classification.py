import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#import umap
from torchvision import models, transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

input_imags = './images/'

img = Image.open(input_imags+'solo_L2_rpw-tds-surv-rswf-e_20210101_V03.cdf.png')
print(img)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # без последнего слоя
model.to(device)
model.eval()
#plt.imshow(img)

#plt.show()