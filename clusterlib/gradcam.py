#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import cv2


def find_medoids(df_features) -> dict:
    """Find the medoid (most central point) for each cluster.

    Parameters
    ----------
    df_features : pd.DataFrame
        Must contain a 'label' column with cluster assignments (−1 = noise).

    Returns
    -------
    medoids : dict mapping cluster_id -> feature vector (np.ndarray)
    """
    feat_col = [col for col in df_features.columns if col != 'label']
    medoids = {}

    for label in df_features['label'].unique():
        if label == -1:
            continue
        mask = df_features['label'] == label
        cluster = df_features.loc[mask, feat_col].values
        dists = pairwise_distances(cluster)
        medoid_idx = dists.sum(axis=1).argmin()
        medoids[label] = cluster[medoid_idx]

    return medoids


def get_images_med(features, medoids: dict) -> dict:
    """Map medoid feature vectors to their source image paths.

    For each cluster finds the sample in features whose feature vector is
    closest (L2) to the medoid vector.

    Parameters
    ----------
    features : pd.DataFrame
        Full feature table including 'dataset_name' and 'oldpath'
        columns alongside feature columns and a 'label' column.
    medoids : dict
        Output of :func:`find_medoids`.

    Returns
    -------
    medoid_images : dict mapping cluster_id -> [dataset_name, image_path]
    """
    excluded = {'dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER',
                'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'oldpath', 'path', 'label'}
    feature_cols = [col for col in features.columns if col not in excluded]

    medoid_images = {}
    for label, medoid_vector in medoids.items():
        mask = features['label'] == label
        cluster = features.loc[mask]
        diffs = np.linalg.norm(cluster[feature_cols].values - medoid_vector, axis=1)
        medoid_row = cluster.iloc[diffs.argmin()]
        medoid_images[label] = [medoid_row['dataset_name'], medoid_row['oldpath']]

    return medoid_images



def build_resnet_backbone():
    """Return a pre-trained ResNet50 without the final classification layer.

    The backbone outputs a (1, 2048, 7, 7) feature map suitable for Grad-CAM.
    Requires torchvision.
    """

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval()
    return backbone


class GradCAM:
    """Forward / backward hook pair for Grad-CAM computation.

    Attach to any intermediate layer of a PyTorch model to capture the
    activations and gradients needed by :func:`compute_gradcam`.

    Parameters
    ----------
    model : nn.Module
        The full model (used during the forward pass).
    target_layer : nn.Module
        The layer to hook (e.g. ``model[-1][-1]`` for the last residual block).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach().clone()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()


def compute_score(feature_map, medoid_vector):
    """Cosine similarity between a pooled feature map and a medoid vector.

    Parameters
    ----------
    feature_map : torch.Tensor of shape (1, 2048, 1, 1)
    medoid_vector : np.ndarray of shape (2048,)

    Returns
    -------
    score : torch.Tensor scalar
    """

    pooled = feature_map.squeeze(-1).squeeze(-1)
    medoid_tensor = torch.tensor(medoid_vector, dtype=torch.float32).unsqueeze(0)
    return F.cosine_similarity(pooled, medoid_tensor)


def compute_gradcam(
    gradcam_obj: GradCAM,
    image_tensor,
    medoid_vector: np.ndarray,
) -> np.ndarray:
    """Generate a Grad-CAM attention map for one image with respect to a medoid.

    Parameters
    ----------
    gradcam_obj : GradCAM
    image_tensor : torch.Tensor of shape (1, 3, 224, 224)
        Normalised image tensor with requires_grad=True.
    medoid_vector : np.ndarray of shape (2048,)

    Returns
    -------
    cam : np.ndarray of shape (7, 7), values in [0, 1]
    """

    gradcam_obj.model.zero_grad()
    feature_map = gradcam_obj.model(image_tensor)
    score = compute_score(feature_map, medoid_vector)
    score.backward()

    gradients  = gradcam_obj.gradients
    activations = gradcam_obj.activations
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def load_image(path: str):
    """Load and preprocess an image for ResNet50.

    Returns
    -------
    img : PIL.Image (original, resized to 224x224 for display)
    tensor : torch.Tensor of shape (1, 3, 224, 224) with requires_grad=True
    """

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    tensor.requires_grad_(True)
    return img, tensor


def overlay_cam(original_image, cam: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original image (50/50 mix).

    Parameters
    ----------
    original_image : PIL.Image
    cam : np.ndarray of shape (7, 7), values in [0, 1]

    Returns
    -------
    overlay : np.ndarray of shape (224, 224, 3), uint8
    """
    img_np = np.array(original_image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.5 * img_np + 0.5 * heatmap)
    return overlay



def visualise_cluster_medoids(
    medoid_images: dict,
    medoids: dict,
    gradcam_obj: GradCAM,
) -> None:
    """Multi-panel figure: medoid image + Grad-CAM overlay for each cluster.

    Two clusters per row; each row contains [medoid, CAM, medoid, CAM].
    """
    n = len(medoid_images)
    n_rows = (n + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

    for idx, (label, image_path) in enumerate(medoid_images.items()):
        row = idx // 2
        col_offset = (idx % 2) * 2

        original_img, image_tensor = load_image(image_path[1])
        cam = compute_gradcam(gradcam_obj, image_tensor, medoids[label])
        overlay = overlay_cam(original_img, cam)

        axes[row, col_offset].imshow(original_img.resize((224, 224)))
        axes[row, col_offset].set_title(f'Cluster {label} — medoid')
        axes[row, col_offset].axis('off')

        axes[row, col_offset + 1].imshow(overlay)
        axes[row, col_offset + 1].set_title(f'Cluster {label} — Grad-CAM')
        axes[row, col_offset + 1].axis('off')

    if n % 2 != 0:
        axes[-1, 2].axis('off')
        axes[-1, 3].axis('off')

    plt.tight_layout()
    plt.show()
