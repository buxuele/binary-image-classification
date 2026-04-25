import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import SaintGeorgeDataset, get_transforms, get_image_paths_and_labels


# Configuration
DATA_DIR = "./data"
POS_DIR = os.path.join(DATA_DIR, "georges")
NEG_DIR = os.path.join(DATA_DIR, "non_georges")
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare data
    pos_data = get_image_paths_and_labels(POS_DIR, 1)
    neg_data = get_image_paths_and_labels(NEG_DIR, 0)
    all_data = pos_data + neg_data
    all_paths = [item[0] for item in all_data]
    all_labels = [item[1] for item in all_data]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    train_tf, val_tf = get_transforms()
    
    train_dataset = SaintGeorgeDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = SaintGeorgeDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Build model and optimizer
    model = build_model().to(DEVICE)
    weight_value = len(neg_data) / len(pos_data)
    pos_weight = torch.tensor([weight_value]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 3. Training loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            corrects += torch.sum(preds == labels.byte()).item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_dataset)
        train_acc = corrects / total
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                val_corrects += torch.sum(preds == labels.byte()).item()
                val_total += labels.size(0)
                
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects / val_total
        
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {time_elapsed:.0f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "resnet18_best_epochs_10.pth")
            print("Model saved!")

    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()