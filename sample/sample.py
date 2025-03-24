import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

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

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    vgg = create_vgg16()
    vgg.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\models/model_weights/vgg16_weights.pth"))
    mobilenet = create_mobilenetv3_large()
    mobilenet.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\models/model_weights/mobilenetv3large_weights.pth"))
    vgg.eval()
    mobilenet.eval()

    models_list = [vgg, mobilenet]
    
    logits_list = []
    with torch.no_grad():
        logits = []
        for model in models_list:
            model.eval()
            output = model(image_tensor)
            logits.append(output)

        logits_avg = torch.mean(torch.stack(logits), dim=0)
        logits_list.append(logits_avg.cpu())

        probs = F.softmax(logits_avg, dim=1)
        predicted_class = torch.argmax(probs, dim=1)

    predicted_class_item = predicted_class.item()
    if predicted_class_item == 0:
        print(f"Predicted Class: {predicted_class_item} - Inme Var")
    else:
        print(f"Predicted Class: {predicted_class_item} - Inme Yok")

# Test both images
image_paths = [
    r"C:\Users\kilca\Desktop\strokeTamami\10036_inmeVar.png",
    r"C:\Users\kilca\Desktop\strokeTamami\10277_inmeYok.png"
]

for image_path in image_paths:
    predict_image(image_path)
