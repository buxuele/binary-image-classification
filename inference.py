import sys
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from pathlib import Path
from dataset import get_transforms   

def load_model(model_path: str, device: torch.device):
    """Load ResNet18 binary classification model."""
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)    
    
    # Load weights (compatible with both common save formats)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded from checkpoint['model_state_dict']")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded full state_dict")
    
    model.to(device)
    model.eval()
    print("Model set to eval mode\n")
    return model


def predict_image(image_path: str, model, device, transform):
    """Run prediction on a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to open image {image_path}: {e}")
        return None
    
    # Preprocess and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).squeeze().item()   # 转为标量
    
    is_george = prob >= 0.5
    class_name = "Saint George (Positive)" if is_george else "Non-George (Negative)"
    
    print(f"Image:       {image_path}")
    print(f"Prediction:  {class_name}")
    print(f"Probability: {prob:.4f}")
    print("-" * 50)
    
    return prob


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python inference.py <image_path>")
        print("Example: python inference.py example_imgs/a1.jpg")
        print("         python inference.py img1.jpg img2.jpg  (multiple supported)")
        sys.exit(1)
    
    model_path = "resnet18_best.pth"
    image_paths = sys.argv[1:]
    
    if not Path(model_path).exists():
        print(f"Error: model file {model_path} not found!")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    # Use validation transforms for inference
    _, val_transform = get_transforms()
    
    # Support multiple image paths
    for img_path in image_paths:
        predict_image(img_path, model, device, val_transform)