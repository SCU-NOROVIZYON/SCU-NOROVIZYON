import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt

data = 52
tryy=3

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(f'/content/drive/MyDrive/ikiSinifliVeri{data}_aug/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'/content/drive/MyDrive/ikiSinifliVeri{data}_aug/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, train_dataset.classes

import torch
import torch.nn as nn

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
        return self.softmax(x), features_512, x


# Eğitim, test, ve görselleştirme fonksiyonları olduğu gibi kalabilir.
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100, patience=15):
    best_acc = 0.0
    best_loss = float('inf')
    early_stopping_counter = 0
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, correct, total = 0, 0, 0
        all_train_labels, all_train_preds = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # Save labels and predictions for F1 score
            all_train_labels.append(labels.cpu().numpy())
            all_train_preds.append(outputs.argmax(1).cpu().numpy())

        # Calculate train accuracy and F1 score
        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        all_train_labels = np.concatenate(all_train_labels)
        all_train_preds = np.concatenate(all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        # Validation step
        model.eval()
        test_loss, correct, total = 0, 0, 0
        all_test_labels, all_test_preds = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                # Save labels and predictions for F1 score
                all_test_labels.append(labels.cpu().numpy())
                all_test_preds.append(outputs.argmax(1).cpu().numpy())

        # Calculate test accuracy and F1 score
        test_acc = correct / total
        test_losses.append(test_loss / len(test_loader))
        all_test_labels = np.concatenate(all_test_labels)
        all_test_preds = np.concatenate(all_test_preds)
        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')

        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {test_loss/len(test_loader):.4f}, Val Acc: {test_acc:.4f}, Val F1: {test_f1:.4f}")

        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"/content/drive/MyDrive/efficientnetb3_aug_model_weights_acc_data{data}_try{tryy}.pth")
            torch.save(model, f"/content/drive/MyDrive/efficientnetb3_aug_model_acc_data{data}_try{tryy}.pth")
            print("accuracy e göre kaydedildi")
            early_stopping_counter = 0  # Test doğruluğu iyileştiğinde sayaç sıfırlanır
        else:
            early_stopping_counter += 1
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f"/content/drive/MyDrive/efficientnetb3_aug_model_weight_loss_data{data}_try{tryy}.pth")
            torch.save(model, f"/content/drive/MyDrive/efficientnetb3_aug_model_loss_data{data}_try{tryy}.pth")
            print("loss a göre kaydedildi")

        if early_stopping_counter >= patience:
            break

    # Plotting loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig(f"/content/drive/MyDrive/efficientnetb3_aug_data{data}_try{tryy}_train_test_looss")
    plt.show()


def test_model(model, test_loader, criterion, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Test sırasında modeli eval moduna al
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

            # Tahminleri ve gerçek etiketleri kaydet
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct_test / total_test
    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Sınıflandırma raporunu oluştur
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def visualize_tsne(model, loader, classes):
    model.eval()
    features_512, features_256, labels = [], [], []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.cuda()
            _, ftrs_512, ftrs_256 = model(inputs)  # Hem 512 hem de 256 katmandan özellik al
            features_512.append(ftrs_512.cpu().numpy())
            features_256.append(ftrs_256.cpu().numpy())
            labels.append(lbls.numpy())

    features_512 = np.concatenate(features_512)
    features_256 = np.concatenate(features_256)
    labels = np.concatenate(labels)

    # Sınıf isimlerini etiketlerden almak için
    label_names = [classes[label] for label in labels]

    # TSNE ile görselleştirme
    tsne = TSNE(n_components=2, perplexity=30, random_state=20)

    # 512 nöronlu katmandan önceki özelliklerin görselleştirilmesi
    transformed_features_512 = tsne.fit_transform(features_512)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_features_512[:, 0], y=transformed_features_512[:, 1], hue=label_names, palette='coolwarm')
    plt.title("t-SNE visualization before the 512 neurons layer")
    plt.savefig(f"/content/drive/MyDrive/efficientnetb3_aug_data{data}_try{tryy}_tsne_before_512")
    plt.show()

    # 256 nöronlu katmandan sonraki özelliklerin görselleştirilmesi
    transformed_features_256 = tsne.fit_transform(features_256)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_features_256[:, 0], y=transformed_features_256[:, 1], hue=label_names, palette='coolwarm')
    plt.title("t-SNE visualization after the 256 neurons layer")
    plt.savefig(f"/content/drive/MyDrive/efficientnetb3_aug_data{data}_try{tryy}_tsne_after_256")
    plt.show()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = get_data_loaders()
efficientnetb3 = models.efficientnet_b3(pretrained=True)
model = CustomEfficientNet(efficientnetb3).to(dev)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)

best_model = torch.load(f"/content/drive/MyDrive/efficientnetb3_aug_model_acc_data{data}_try{tryy}.pth", weights_only = False).to(dev)

test_model(best_model, test_loader, criterion, classes)
visualize_tsne(best_model, test_loader, classes)
