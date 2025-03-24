import timm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def create_model(num_classes=2, device='cuda'):
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
        nn.Softmax(dim=1)  
    )

    return model.to(device)

def extract_features(model, dataloader, device):
    model.eval()
    features_512, features_256, labels_list = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

          
            x = model.classif[0](model.avg_pool(model.forward_features(inputs)))  
            x = model.classif[1](x)  
            x = model.classif[2](x) 
            x = model.classif[3](x)  

            features_512.append(x.cpu().numpy())
            x = model.classif[4](x)  
            x = model.classif[5](x) 
            x = model.classif[6](x)  

            features_256.append(x.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    features_512 = np.concatenate(features_512, axis=0)
    features_256 = np.concatenate(features_256, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    return features_512, features_256, labels_list

def plot_tsne(tsne_results, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", alpha=0.7)
    plt.title(title)
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.legend(title="Classes", loc="best")
    plt.show()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    base_path =" data/"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train veya Test klasörü bulunamadı: {train_path} veya {test_path}")

    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0001
    num_classes = 2  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 15
    best_test_loss = float('inf')
    best_test_accuracy = 0.0
    early_stopping_counter = 0  
    
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = create_model(num_classes=num_classes, device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_model_path_acc = "inception_resnet_v2_model_acc.pth"
    best_weights_path_acc = "inception_resnet_v2_weights_acc.pth"

    best_model_path_loss = "inception_resnet_v2_model_loss.pth"
    best_weights_path_loss = "inception_resnet_v2_weights_loss.pth"

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

    print("Confusion Matrix:")
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print(cm)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_test_labels, all_test_preds))
    
    print("\nModel Sonuçları:")
    print(f"Test Accuracy: {accuracy_score(all_test_labels, all_test_preds) * 100:.2f}%")
    print(f"Test Precision: {precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=1) * 100:.2f}%")
    print(f"Test Recall: {recall_score(all_test_labels, all_test_preds, average='weighted') * 100:.2f}%")
    print(f"Test F1 Score: {f1_score(all_test_labels, all_test_preds, average='weighted') * 100:.2f}%")
    
    features_512, features_256, labels = extract_features(model, test_loader, device)

    
    tsne_512 = TSNE(n_components=2, random_state=42).fit_transform(features_512)
    tsne_256 = TSNE(n_components=2, random_state=42).fit_transform(features_256)

    
    plot_tsne(tsne_512, labels, "TSNE Visualization of 512-D Features")
    plot_tsne(tsne_256, labels, "TSNE Visualization of 256-D Features")
