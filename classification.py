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

img = Image.open(input_imags+'solo_L2_rpw-tds-surv-rswf-e_20210101_V03.cdf.png').convert('RGB')
print(img)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img)

#print(img_tensor)
print(img_tensor.shape)

batch = img_tensor.unsqueeze(0).to(device)
print(batch.shape)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # without the last layer
model.to(device)
model.eval()
features = []
with torch.no_grad():
    feat = model(batch).squeeze().cpu().numpy()
print(feat.shape)

features.append(feat)
features = np.array(features)
print(features.shape)

#plt.imshow(img_tensor)

#plt.show()