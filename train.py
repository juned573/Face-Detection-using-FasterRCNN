import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import FaceDataset  # Make sure this file exists and is correct

print(" Training script started...")

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset and dataloader
dataset = FaceDataset("WIDER_train/images", "train_annotations.json")
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

# Replace classification head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

# Training loop
model.train()
for epoch in range(5):
    print(f" Starting epoch {epoch+1}")
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches with no boxes
        if not all('boxes' in t and t['boxes'].numel() > 0 for t in targets):
            print(f" Skipping batch {i+1}: no valid bounding boxes.")
            continue

        try:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f" Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

        except Exception as e:
            print(f" Error at epoch {epoch+1}, batch {i+1}: {e}")
            continue

# Saving the trained model
torch.save(model.state_dict(), "fasterrcnn_face.pth")
print("Model saved")
