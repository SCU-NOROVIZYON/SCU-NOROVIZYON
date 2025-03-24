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

def create_inception_resnet_v2(num_classes=2):
    model = timm.create_model("inception_resnet_v2", pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    in_features = model.classif.in_features
    model.classif = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
          
    )
    return model.to(device)

class CustomDenseNet(nn.Module):
    def __init__(self, base_model):
        super(CustomDenseNet, self).__init__()

        # **Tüm katmanları eğitime aç**
        for param in base_model.parameters():
            param.requires_grad = True  # Tüm katmanları eğitilebilir yap

        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Son katmanı çıkar
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 512)  # DenseNet'in son katmanındaki özellik sayısı 1024
        self.dropout1 = nn.Dropout(0.5)  # Dropout ekle
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)  # Dropout ekle
        self.fc3 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        features_512 = self.fc1(x)
        x = self.dropout1(features_512)  # Dropout sonrası
        x = self.fc2(x)
        x = self.dropout2(x)  # Dropout sonrası
        x = self.fc3(x)
        return x

class CustomResNet50(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet50, self).__init__()
        
        # **Tüm katmanları eğitime aç**
        for param in base_model.parameters():
            param.requires_grad = True  # Tüm katmanları eğitilebilir yap

        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected katmanları
        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(0.5)  # İlk fully connected katmandan sonra dropout
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)  # İkinci fully connected katmandan sonra dropout
        self.fc3 = nn.Linear(256, 2)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.base(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        features_512 = self.fc1(x)
        features_512 = self.dropout1(features_512)  # Dropout uygula
        
        x = self.fc2(features_512)
        x = self.dropout2(x)  # Dropout uygula
        
        x = self.fc3(x)
        return x

class CustomEfficientNet(nn.Module):

    def __init__(self, base_model):
        super(CustomEfficientNet, self).__init__()

        # Tüm katmanları eğitime aç
        for param in base_model.parameters():
            param.requires_grad = True

        self.base = nn.Sequential(*list(base_model.children())[:-2])  # Son FC katmanını çıkar
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1536, 512)  # EfficientNet-B3 için çıkış özellik sayısı 1536
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)  # 2 sınıflı problem varsayıldı
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        features_512 = self.fc1(x)
        x = self.dropout1(features_512)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_e = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(r'C:/Users/kilca/Desktop/strokeTamami/ikiSinifliVeri52_aug/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_dataset_e = datasets.ImageFolder(r'C:/Users/kilca/Desktop/strokeTamami/ikiSinifliVeri52_aug/test', transform=transform_e)
test_loader_e = DataLoader(test_dataset_e, batch_size=32, shuffle=False)

vgg = create_vgg16()
vgg.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\weights/best_weights_vgg16_model_acc_2classes.pth"))
mobilenet = create_mobilenetv3_large()
mobilenet.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\weights/best_weights_mobilenetv3large_model_acc_2classes.pth"))
inception_resnet = create_inception_resnet_v2()
inception_resnet.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\weights/best_weights_resnet-inceptionv2_model_acc_2classes.pth"))
base_densenet = models.densenet121(pretrained=True)
densenet = CustomDenseNet(base_densenet).to(device)
densenet.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\weights\densenet121_aug_model_weights_acc_data52_try2.pth"))
base_efficientnet = models.efficientnet_b3(pretrained=True)
efficientnet = CustomEfficientNet(base_efficientnet).to(device)
efficientnet.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\weights\efficientnetb3_aug_model_weights_acc_data52_try2.pth"))
base_resnet = models.resnet50(pretrained=True)
resnet = CustomResNet50(base_resnet).to(device)
resnet.load_state_dict(torch.load(r"C:\Users\kilca\Desktop\weights\resnet50_aug_model_weights_acc_data52_try1.pth"))
vgg.eval()
mobilenet.eval()
inception_resnet.eval()
densenet.eval() 
efficientnet.eval()
resnet.eval()


logits_list = []
true_labels = []
predicted_labels = []

models = [vgg,mobilenet]

with torch.no_grad():
    for (images, labels), (images_e, _) in zip(test_loader, test_loader_e):  
        images, images_e = images.to(device), images_e.to(device)
        true_labels.extend(labels.cpu().numpy())  

        logits = []

        for model in models[:5]:
            logits.append(model(images))

        logits.append(models[5](images_e))

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