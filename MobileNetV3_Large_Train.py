import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def create_model(num_classes=2, device='cuda'):
    # MobileNetV3 modelini yükle
    mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    for param in mobilenet_v3_large.parameters():
        param.requires_grad = True  # Tüm parametrelerin eğitilmesine izin ver

    # Global average pooling
    mobilenet_v3_large.avgpool = nn.AdaptiveAvgPool2d(1)

    # Son katmanları ayarlama
    mobilenet_v3_large.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(960, 512),  # 960 → 512, doğru çıkış boyutunu kullan
        nn.ReLU(),
        nn.BatchNorm1d(512),  # Batch Normalization
        nn.Dropout(0.5),  # Dropout
        nn.Linear(512, 256),  # 512 → 256
        nn.ReLU(),
        nn.Linear(256, num_classes),  # 256 → 2 (çıkış katmanı)
        nn.Softmax(dim=1)  # Softmax
    )

    return mobilenet_v3_large.to(device)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    # Kullanıcıdan gelen veri yolu
    base_path = r"C:\Users\Monster\Desktop\ikiSinifliVeri_seed32_aug"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train veya Test klasörü bulunamadı: {train_path} veya {test_path}")

    batch_size = 32
    num_epochs = 125
    learning_rate = 0.0001
    num_classes = 2  # 2 sınıf (Hastalıklı ve Sağlıklı)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 15
    best_test_loss = float('inf')
    best_test_accuracy = 0.0
    early_stopping_counter = 0  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Modeli oluştur
    model = create_model(num_classes=num_classes, device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_model_path_acc = "best_mobilenetv3_model_acc_2classes.pth"
    best_weights_path_acc = "best_weights_mobilenetv3_model_acc_2classes.pth"

    best_model_path_loss = "best_mobilenetv3_model_loss_2classes.pth"
    best_weights_path_loss = "best_weights_mobilenetv3_model_loss_2classes.pth"

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    def visualize_with_tsne(model, data_loader, feature_layer):
        model.eval()
        
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                x = inputs
                for name, layer in model.named_children():
                    x = layer(x)
                    if name == feature_layer:
                        features.append(x.view(x.size(0), -1).cpu().numpy()) 
                        labels.extend(targets.cpu().numpy())
                        break
        
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)
        
        tsne = TSNE(n_components=2, random_state=42)
        tsne_features = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='jet', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f't-SNE visualization after {feature_layer} layer')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        all_train_labels, all_train_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=1)
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        test_loss, test_correct = 0.0, 0
        all_test_labels, all_test_preds = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)
        test_precision = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=1)
        test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        torch.cuda.empty_cache()
        gc.collect()

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), best_weights_path_acc)
            torch.save(model, best_model_path_acc)
            print("Accuracy'ye göre model kaydedildi.")
            early_stopping_counter = 0  

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_weights_path_loss)
            torch.save(model, best_model_path_loss)
            print("Loss'a göre model kaydedildi.")
            early_stopping_counter = 0  

        if test_accuracy <= best_test_accuracy and test_loss >= best_test_loss:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}")

        if early_stopping_counter >= patience:
            print(f"Erken durdurma tetiklendi. Epoch {epoch+1} sonunda eğitim durduruluyor.")
            break

        scheduler.step(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, "
              f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    model.load_state_dict(torch.load(best_weights_path_acc))  
    model.eval()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()

    plt.show()

    all_test_labels, all_test_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    cm = confusion_matrix(all_test_labels, all_test_preds)
    print(f"\nConfusion Matrix:\n{cm}")
   
    print("\nClassification Report:\n", classification_report(all_test_labels, all_test_preds))
    
    print("\nModel Sonuçları:")
    print(f"Test Accuracy: {accuracy_score(all_test_labels, all_test_preds) * 100:.2f}%")
    print(f"Test Precision: {precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=1) * 100:.2f}%")
    print(f"Test Recall: {recall_score(all_test_labels, all_test_preds, average='weighted') * 100:.2f}%")
    print(f"Test F1 Score: {f1_score(all_test_labels, all_test_preds, average='weighted') * 100:.2f}%")

    print("\nEğitim tamamlandı. En iyi model kaydedildi.")
   
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hastalıklı', 'Sağlıklı'], yticklabels=['Hastalıklı', 'Sağlıklı'])
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Görselleştirmeyi yapma
    visualize_with_tsne(model, test_loader, 'features')  # 512'den sonra
    visualize_with_tsne(model, test_loader, 'classifier')  # 256'dan sonra
