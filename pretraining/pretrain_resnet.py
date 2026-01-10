'''
using 20~ minutes of footage to pretrain Resnet 18 before offline RL stage
'''

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import random

# --- CONFIGURATION ---
VIDEO_PATH = "input.mp4"
BATCH_SIZE = 32
LATENT_DIM = 512
K_OFFSET = 4  # Frames ahead for Positive sample (at 20fps, ~200ms)
IMG_SIZE = (640, 640)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET ---
class RustTemporalDataset(Dataset):
    def __init__(self, video_path, k_offset=4, transform=None):
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.k_offset = k_offset
        self.transform = transform

    def __len__(self):
        # Ensure we have room for the future positive frame
        return self.total_frames - self.k_offset - 1

    def get_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret: return torch.zeros(3, *IMG_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(frame)
        return frame

    def __getitem__(self, idx):
        # Anchor (t)
        anchor = self.get_frame(idx)
        # Positive (t + k)
        positive = self.get_frame(idx + self.k_offset)
        # Negative (Random frame)
        neg_idx = random.randint(0, self.total_frames - self.k_offset - 1)
        while abs(neg_idx - idx) < 100: # Ensure negative isn't too close in time
            neg_idx = random.randint(0, self.total_frames - self.k_offset - 1)
        negative = self.get_frame(neg_idx)
        
        return anchor, positive, negative
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel

        # Create a grid of coordinates [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, self.height),
            torch.linspace(-1, 1, self.width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, x):
        # x shape: [batch, channel, h, w]
        batch_size = x.size(0)
        x = x.view(batch_size, self.channel, -1)
        
        # Apply softmax over the spatial dimensions (h*w)
        softmax_attention = F.softmax(x, dim=-1) # [batch, channel, h*w]
        
        # Calculate expected X and Y coordinates
        expected_x = torch.sum(softmax_attention * self.pos_x, dim=-1)
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=-1)
        
        # Stack coordinates: [batch, channel * 2]
        return torch.cat([expected_x, expected_y], dim=-1)
    
# --- MODEL ---
class ResNetSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Remove the original pooling and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # At 640x640 input, ResNet-18's final feature map is 20x20
        # and has 512 channels.
        self.spatial_softmax = SpatialSoftmax(20, 20, 512)
        
        # Output will be 512 channels * 2 coordinates = 1024 features
        self.projector = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.spatial_softmax(x)
        return self.projector(x)
    
# --- LOSS FUNCTION (InfoNCE) ---
def contrastive_loss(anchor, positive, negative, temperature=0.07):
    # Normalize features
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    
    # Positive logit: cos similarity(anchor, positive)
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
    # Negative logit: cos similarity(anchor, negative)
    neg_sim = torch.sum(anchor * negative, dim=-1) / temperature
    
    # We want pos_sim to be large and neg_sim to be small
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(DEVICE) # Label 0 is the positive
    
    return F.cross_entropy(logits, labels)

# --- TRAINING LOOP ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = RustTemporalDataset(VIDEO_PATH, k_offset=K_OFFSET, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = ResNetTPC().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for i, (a, p, n) in enumerate(loader):
        a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
        
        z_a = model(a)
        z_p = model(p)
        z_n = model(n)
        
        loss = contrastive_loss(z_a, z_p, z_n)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}] Step [{i}] Loss: {loss.item():.4f}")

# Save the backbone for RL phase
torch.save(model.backbone.state_dict(), "rust_backbone.pth")