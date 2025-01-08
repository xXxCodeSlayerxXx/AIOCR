# notes to self:

# to use this on compute server:
#     ssh wmr440@3.compute.vu.nl and login
#     cd /local/data/wmr440/files
#     scp /home/william/Documents/personal/aibenchmark.py wmr440@3.compute.vu.nl:/local/data/wmr440/files


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(set(''.join(labels)))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to tensor
        label = torch.tensor([self.char_to_idx[c] for c in self.labels[idx]])
        
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, num_chars):
        super(SimpleCNN, self).__init__()
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 10, 512),  # Adjust size based on your input dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_chars)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Example usage
def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # Resize images to fixed size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset and dataloader
    # Replace with your actual data paths and labels
    dataset = OCRDataset(image_paths=['path1.png', 'path2.png'], 
                        labels=['text1', 'text2'],
                        transform=transform)
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = SimpleCNN(num_chars=len(dataset.char_to_idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer)

if __name__ == "__main__":
    main()