import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device: {torch.backends.mps.is_available()}")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

torch.manual_seed(42)
if device.type == 'mps':
    torch.mps.manual_seed(42)

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),  
    transforms.RandomRotation(25),         
    transforms.ColorJitter(brightness=0.15, contrast=0.15), 
    transforms.GaussianBlur(kernel_size=(3, 7)), 
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

data_dir = "Plant_leave_diseases_dataset_without_augmentation"
full_dataset = datasets.ImageFolder(root=data_dir)

train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

BATCH_SIZE = 16  
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,  
    pin_memory=True
)

class_names = full_dataset.classes

class MPSOptimizedModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.base_model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
            
        in_features = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),  
            nn.ReLU(),                     
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        self.to(device)

    def forward(self, x):
        return self.base_model(x)

model = MPSOptimizedModel(len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(  
    model.parameters(), 
    lr=0.001,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    'min', 
    patience=2,
    factor=0.5
)

def mps_train(num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_mps.pth')
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2%}\n")
    
    return history

history = mps_train(num_epochs=10)

if device.type == 'mps':
    torch.mps.empty_cache()