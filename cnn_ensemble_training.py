# CNN 앙상블 학습 전체 코드 + 시각화 10종 포함

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# RTX 4050 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB") 
    # CUDA 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 활성화
    torch.backends.cudnn.allow_tf32 = True  # TensorFloat-32 활성화

# RTX 4050에 최적화된 하이퍼파라미터
IMG_SIZE = 224
BATCH_SIZE = 128  # RTX 4050 6GB VRAM에 최적화
EPOCHS = 50
LEARNING_RATE = 0.002
NUM_WORKERS = 4  # 데이터 로딩 최적화
WEIGHT_DECAY = 0.01

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 모델 저장 디렉토리 설정
SAVE_DIR = 'saved_models'
RESULTS_DIR = 'results'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"모델 저장 경로: {os.path.abspath(SAVE_DIR)}")
print(f"결과 저장 경로: {os.path.abspath(RESULTS_DIR)}")

# Mixup 함수 개선
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 데이터셋 클래스
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images, self.labels = [], []

        print(f"\n데이터 로딩 중... 디렉토리: {root_dir}")
        for class_name in tqdm(self.classes, desc="클래스 로딩"):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        print(f"총 {len(self.images)}개의 이미지 로드 완료\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# 모델 구조 개선
class BaseCNN(nn.Module):
    def __init__(self, channels, dropout, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),  # inplace 연산으로 메모리 효율화
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # 특징맵 드롭아웃 추가
            
            # 두 번째 블록
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # 세 번째 블록
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # 네 번째 블록
            nn.Conv2d(channels[2], channels[3], 3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[3], 1024),  # 더 큰 은닉층
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # 드롭아웃 비율 조정
            nn.Linear(512, num_classes)
        )
        
        # 가중치 초기화 추가
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes):
    if model_name == 'A':
        return BaseCNN([64, 128, 256, 512], 0.4, num_classes)
    elif model_name == 'B':
        return BaseCNN([96, 192, 384, 768], 0.5, num_classes)  # 더 큰 모델
    elif model_name == 'C':
        return BaseCNN([48, 96, 192, 384], 0.3, num_classes)

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    print(f"\n=== Training Model {model_name} ===")
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []
    best_val_acc = 0.0
    
    # 모델별 저장 디렉토리 생성
    model_save_dir = os.path.join(SAVE_DIR, f'model_{model_name}')
    os.makedirs(model_save_dir, exist_ok=True)
    
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Results:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 모델 저장 (매 에폭)
        epoch_save_path = os.path.join(model_save_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_acc': train_accuracy,
            'val_acc': val_accuracy,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, epoch_save_path)
        
        # 최고 성능 모델 저장
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            print(f"새로운 최고 성능 모델 저장 (검증 정확도: {val_accuracy:.2f}%)")
            print(f"저장 경로: {best_model_path}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_accuracy,
                'train_accs': train_acc_list,
                'val_accs': val_acc_list,
                'train_losses': train_loss_list,
                'val_losses': val_loss_list
            }, best_model_path)
        
        # 학습 진행 상황 저장
        progress_path = os.path.join(RESULTS_DIR, f'model_{model_name}_progress.pth')
        torch.save({
            'current_epoch': epoch,
            'best_val_acc': best_val_acc,
            'train_accs': train_acc_list,
            'val_accs': val_acc_list,
            'train_losses': train_loss_list,
            'val_losses': val_loss_list
        }, progress_path)
        
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    return train_acc_list, val_acc_list, train_loss_list, val_loss_list

def main():
    print(f"Training device: {device}")
    set_seed()
    
    # 데이터셋 및 데이터로더 설정
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\n데이터셋 로딩 중...")
    train_set = FoodDataset('./food_dataset/train', transform)
    val_set = FoodDataset('./food_dataset/val', val_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"\n데이터셋 정보:")
    print(f"총 클래스 수: {len(train_set.classes)}")
    print(f"학습 데이터 수: {len(train_set)}")
    print(f"검증 데이터 수: {len(val_set)}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 각 모델 학습
    models_results = {}
    for model_name in ['A', 'B', 'C']:
        print(f"\n{'='*50}")
        print(f"모델 {model_name} 학습 시작")
        print(f"{'='*50}")
        
        model = get_model(model_name, len(train_set.classes))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LEARNING_RATE,
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        train_acc_list, val_acc_list, train_loss_list, val_loss_list = train_model(
            model_name, model, train_loader, val_loader, 
            criterion, optimizer, scheduler, EPOCHS
        )
        
        models_results[model_name] = {
            'train_accs': train_acc_list,
            'val_accs': val_acc_list,
            'train_losses': train_loss_list,
            'val_losses': val_loss_list
        }
        
        print(f"\n모델 {model_name} 학습 완료")
        print(f"최종 검증 정확도: {val_acc_list[-1]:.2f}%")
        
        torch.cuda.empty_cache()
    
    # 전체 결과 저장
    final_results_path = os.path.join(RESULTS_DIR, 'final_results.pth')
    torch.save(models_results, final_results_path)
    print(f"\n최종 결과가 저장되었습니다: {final_results_path}")

if __name__ == "__main__":
    main()