# Self-Supervised Learning with BYOL and ResNet50 on MIMIC Dataset

This repository contains the implementation of a self-supervised learning framework using the Bootstrap Your Own Latent (BYOL) method with a ResNet18 backbone on the MIMIC medical imaging dataset. The code includes data loading, model definition, training, and evaluation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Description](#code-description)
  - [Configuration](#configuration)
  - [Data Loading](#data-loading)
  - [Model Definition](#model-definition)
  - [BYOL Loss Function](#byol-loss-function)
  - [Data Transformations](#data-transformations)
  - [Training and Evaluation](#training-and-evaluation)
  - [Plotting ROC AUC Scores](#plotting-roc-auc-scores)
- [Logging and Model Checkpoint](#logging-and-model-checkpoint)

## Installation

To run this code, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- PyYAML
- tqdm
- matplotlib
- numpy

Install the dependencies using pip:

```bash
pip install torch torchvision scikit-learn pyyaml tqdm matplotlib numpy
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Update the `config1.yaml` file with the appropriate paths and parameters for your dataset and training setup.

3. Run the training script:
   ```bash
   python train.py
   ```

## Code Description

### Configuration

The configuration is loaded from `config1.yaml`, which sets parameters such as the dataset path, batch size, number of epochs, and learning rate.

```yaml
# config1.yaml
data_path: "path/to/dataset"
batch_size: 32
num_epochs: 10
learning_rate: 0.001
data_pct: 100
```

### Data Loading

A custom `DataLoader` class is used to load the MIMIC dataset, returning PyTorch DataLoader objects for training, validation, and testing.

```python
# Data loading
sys.path.append('data/data_loader.py')
from data import DataLoader
data_ins = DataLoader(config)
train_loader, valid_loader, test_loader = data_ins.GetMimicDataset()
```

### Model Definition

The `BYOL` class defines the BYOL model with a ResNet18 backbone. The model includes an online and a target network, updated using a moving average.

```python
class BYOL(nn.Module):
    def __init__(self, base_encoder, hidden_dim=4096, projection_dim=256, num_classes=15, moving_average_decay=0.99):
        # Model architecture and initialization
        ...
        
    def forward(self, x1, x2=None):
        # Forward pass
        ...
        
    def update_target_network(self):
        # Update target network
        ...
```

### BYOL Loss Function

The custom BYOL loss function computes the cosine similarity between the online and target projections.

```python
def byol_loss(p1, p2, z1, z2):
    # BYOL loss calculation
    ...
```

### Data Transformations

Data augmentation techniques are applied to input images to create positive pairs for self-supervised learning.

```python
byol_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(30),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Training and Evaluation

The training process includes two phases: pre-training using the BYOL objective and fine-tuning the classifier with a binary cross-entropy loss. ROC AUC scores are computed for validation.

```python
# BYOL Pre-training
for epoch in range(num_epochs):
    byol_model.train()
    for images, _ in tqdm(train_loader):
        images = images.to(device)
        images_transformed = torch.stack([byol_transforms(to_pil_image(img.cpu())) for img in images]).to(device)
        optimizer.zero_grad()
        online_proj_one, online_proj_two, target_proj_one, target_proj_two = byol_model(images, images_transformed)
        loss = criterion(online_proj_one, online_proj_two, target_proj_one, target_proj_two)
        loss.backward()
        optimizer.step()
        byol_model.update_target_network()

# Classifier Fine-tuning
for epoch in range(num_epochs):
    byol_model.train()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = byol_model(images)
        loss = classification_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    byol_model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = byol_model(images)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    roc_auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds), average=None)
```

### Plotting ROC AUC Scores

ROC AUC scores for each class are plotted over epochs to visualize performance.

```python
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot([roc_auc[i] for roc_auc in roc_auc_scores], label=f'Class {i}')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('ROC AUC Scores per Epoch')
plt.legend()
plt.grid(True)
plt.show()
```

## Logging and Model Checkpoint

Training progress and metrics are logged to `training.log`, and the trained model is saved as `byol_model.pth`.

```python
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Save the trained model
torch.save(byol_model.state_dict(), "byol_model.pth")
```

---

Feel free to use and modify this README description as needed for your GitHub repository.
