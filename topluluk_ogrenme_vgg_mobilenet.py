import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import timm
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_vgg16(num_classes=2):
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in vgg16.features.parameters():
        param.requires_grad = True
    vgg16.avgpool = nn.AdaptiveAvgPool2d(1)
    vgg16.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        
    )
    return vgg16.to(device)

def create_mobilenetv3_large(num_classes=2):
    mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    for param in mobilenet_v3_large.parameters():
        param.requires_grad = True
    mobilenet_v3_large.avgpool = nn.AdaptiveAvgPool2d(1)
    mobilenet_v3_large.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(960, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
        
    )
    return mobilenet_v3_large.to(device)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_dataset = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


vgg = create_vgg16()
vgg.load_state_dict(torch.load("model_weights/vgg16_weights"))
mobilenet = create_mobilenetv3_large()
mobilenet.load_state_dict(torch.load("model_weights/mobilenet_weights"))
vgg.eval()
mobilenet.eval()



logits_list = []
true_labels = []
predicted_labels = []

models = [vgg, mobilenet]

with torch.no_grad():
    for images, labels in test_loader:  
        images = images.to(device)
        true_labels.extend(labels.cpu().numpy())  

        logits = []

        for model in models:
            logits.append(model(images))

        

        logits_avg = torch.mean(torch.stack(logits), dim=0)
        logits_list.append(logits_avg.cpu()) 

        probs = F.softmax(logits_avg, dim=1)
        predicted_labels.extend(torch.argmax(probs, dim=1).cpu().numpy())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
