import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SaintGeorgeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_path}. Error: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    """Return train and val/test data augmentation transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transforms, val_test_transforms

def get_image_paths_and_labels(folder_path, label):
    """Collect all image paths from a folder and assign the given label."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [(os.path.join(folder_path, f), label) for f in os.listdir(folder_path) 
            if f.lower().endswith(valid_extensions)]