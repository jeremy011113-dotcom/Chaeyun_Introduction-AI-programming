# CELL
!pip install torch torchvision matplotlib seaborn scikit-learn

# CELL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 재현성 시드
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch: {torch.__version__}')
print(f'Device : {device}')

# CELL
# ─── 전처리 정의 ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 공식 통계값
])

# ─── 데이터셋 다운로드 & 로드 ─────────────────────────────────────────────
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f'학습 데이터: {len(train_dataset):,}개')
print(f'테스트 데이터: {len(test_dataset):,}개')
print(f'이미지 크기: {train_dataset[0][0].shape}  (채널×높이×너비)')
print(f'클래스: {train_dataset.classes}')

# CELL
# ─── 샘플 이미지 시각화 ───────────────────────────────────────────────────
fig, axes = plt.subplots(4, 10, figsize=(16, 7))
fig.patch.set_facecolor('#0d1b2a')
fig.suptitle('MNIST 샘플 이미지 (숫자 0~9, 각 4개)', color='white', fontsize=14)

# 각 숫자별 샘플 찾기
samples = {i: [] for i in range(10)}
for img, label in train_dataset:
    if len(samples[label]) < 4:
        samples[label].append(img)
    if all(len(v) == 4 for v in samples.values()):
        break

for col, digit in enumerate(range(10)):
    for row in range(4):
        ax = axes[row][col]
        img = samples[digit][row].squeeze().numpy()
        ax.imshow(img, cmap='hot', vmin=-1, vmax=2)
        ax.axis('off')
        if row == 0:
            ax.set_title(str(digit), color='#00c9a7', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# 픽셀 분포 히스토그램
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0d1b2a')
for ax in [ax1, ax2]:
    ax.set_facecolor('#162236')
    ax.tick_params(colors='#7fa8c4')
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e3a54')

# 원본 픽셀값 분포
raw = datasets.MNIST('./data', train=True, download=False,
                     transform=transforms.ToTensor())
pixels_raw = torch.stack([raw[i][0] for i in range(1000)]).numpy().flatten()
ax1.hist(pixels_raw, bins=50, color='#00c9a7', alpha=0.8, edgecolor='#0d1b2a')
ax1.set_title('정규화 전 픽셀 분포 [0,1]', color='white')
ax1.set_xlabel('픽셀 값', color='#7fa8c4')

# 정규화 후 픽셀값 분포
pixels_norm = torch.stack([train_dataset[i][0] for i in range(1000)]).numpy().flatten()
ax2.hist(pixels_norm, bins=50, color='#ffb74d', alpha=0.8, edgecolor='#0d1b2a')
ax2.set_title('정규화 후 픽셀 분포', color='white')
ax2.set_xlabel('픽셀 값 (표준화)', color='#7fa8c4')

plt.suptitle('MNIST 픽셀 분포', color='white', fontsize=13)
plt.tight_layout()
plt.show()

# CELL
class MnistCNN(nn.Module):
    """
    MNIST 손글씨 분류를 위한 CNN 모델
    
    Architecture:
        Conv Block 1: Conv(1→32) → BN → ReLU → MaxPool
        Conv Block 2: Conv(32→64) → BN → ReLU → MaxPool
        Classifier:   FC(1600→128) → Dropout → FC(128→10)
    """
    def __init__(self, dropout_rate=0.5):
        super(MnistCNN, self).__init__()

        # ── 합성곱 블록 1 ──────────────────────────────────────────────
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),  # 1×28×28 → 32×28×28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32×28×28 → 32×14×14
        )

        # ── 합성곱 블록 2 ──────────────────────────────────────────────
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),  # 32×14×14 → 64×14×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64×14×14 → 64×7×7
        )

        # ── 완전연결 분류기 ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),             # 64×7×7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block1(x)   # Conv Block 1
        x = self.conv_block2(x)   # Conv Block 2
        x = self.classifier(x)   # Classifier
        return x                  # raw logits (CrossEntropyLoss 사용)

    def get_feature_maps(self, x):
        """Feature Map 시각화를 위해 중간 출력 반환"""
        feat1 = self.conv_block1(x)
        feat2 = self.conv_block2(feat1)
        return feat1, feat2


model = MnistCNN(dropout_rate=0.5).to(device)
print(model)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\n총 파라미터 : {total_params:,}개')
print(f'학습 파라미터: {train_params:,}개')

# CELL
# ─── 각 레이어의 출력 크기 확인 ──────────────────────────────────────────
dummy = torch.zeros(1, 1, 28, 28).to(device)
print('레이어별 출력 크기 추적:')
print(f'  Input         : {dummy.shape}')
out1 = model.conv_block1(dummy)
print(f'  Conv Block 1  : {out1.shape}')
out2 = model.conv_block2(out1)
print(f'  Conv Block 2  : {out2.shape}')
flat = out2.view(1, -1)
print(f'  Flatten       : {flat.shape}')
out_final = model(dummy)
print(f'  Output (logits): {out_final.shape}')

# CELL
# ─── DataLoader ──────────────────────────────────────────────────────────
BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f'배치 크기: {BATCH_SIZE}')
print(f'학습 배치 수: {len(train_loader)}')
print(f'테스트 배치 수: {len(test_loader)}')

# 첫 배치 확인
images, labels = next(iter(train_loader))
print(f'\n첫 배치 이미지 형태: {images.shape}  (배치×채널×높이×너비)')
print(f'첫 배치 레이블 형태: {labels.shape}')

# CELL
def train_one_epoch(model, loader, criterion, optimizer):
    """1 에폭 학습 함수."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """평가 함수. (그래디언트 계산 없음)"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

# CELL
# ─── 하이퍼파라미터 ───────────────────────────────────────────────────────
NUM_EPOCHS    = 15
LEARNING_RATE = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 학습률 스케줄러: 5 epoch마다 lr × 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

print(f'학습 시작 (Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, lr: {LEARNING_RATE})')
print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>9} | {"Test Loss":>9} | {"Test Acc":>8} | {"LR":>8}')
print('-' * 65)

best_test_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['test_loss'].append(te_loss)
    history['test_acc'].append(te_acc)

    # 최고 정확도 모델 저장
    if te_acc > best_test_acc:
        best_test_acc = te_acc
        torch.save(model.state_dict(), 'mnist_cnn_best.pth')
        mark = ' ⭐'
    else:
        mark = ''

    print(f'{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | '
          f'{te_loss:>9.4f} | {te_acc*100:>7.2f}%{mark} | {current_lr:.6f}')

print(f'\n✅ 학습 완료!  최고 Test Accuracy: {best_test_acc*100:.2f}%')

# CELL
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0d1b2a')
epochs = range(1, NUM_EPOCHS + 1)

for ax in axes:
    ax.set_facecolor('#162236')
    ax.tick_params(colors='#7fa8c4')
    ax.set_xlabel('Epoch', color='#7fa8c4')
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e3a54')
    ax.grid(alpha=0.2, color='#1e3a54')

# 손실 곡선
axes[0].plot(epochs, history['train_loss'], color='#00c9a7', lw=2, label='Train')
axes[0].plot(epochs, history['test_loss'],  color='#ffb74d', lw=2, label='Test', linestyle='--')
axes[0].set_title('Loss Curve', color='white', fontsize=13)
axes[0].set_ylabel('CrossEntropyLoss', color='#7fa8c4')
axes[0].legend(facecolor='#162236', labelcolor='white')

# 정확도 곡선
axes[1].plot(epochs, [a*100 for a in history['train_acc']], color='#00c9a7', lw=2, label='Train')
axes[1].plot(epochs, [a*100 for a in history['test_acc']],  color='#ffb74d', lw=2, label='Test', linestyle='--')
axes[1].axhline(y=99, color='#69f0ae', lw=1, linestyle=':', alpha=0.7, label='99% 목표')
axes[1].set_title('Accuracy Curve', color='white', fontsize=13)
axes[1].set_ylabel('Accuracy (%)', color='#7fa8c4')
axes[1].set_ylim(90, 101)
axes[1].legend(facecolor='#162236', labelcolor='white')

plt.suptitle('MNIST CNN 학습 곡선', color='white', fontsize=14)
plt.tight_layout()
plt.show()

# CELL
# ─── 최고 모델 로드 후 최종 평가 ─────────────────────────────────────────
model.load_state_dict(torch.load('mnist_cnn_best.pth', map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

y_pred = np.array(all_preds)
y_true = np.array(all_labels)
final_acc = (y_pred == y_true).mean()

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0d1b2a')
ax.set_facecolor('#162236')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10),
            ax=ax, linewidths=0.5, linecolor='#0d1b2a',
            annot_kws={'size':11})
ax.set_title(f'Confusion Matrix  (Test Accuracy: {final_acc*100:.2f}%)',
             color='white', fontsize=13, pad=12)
ax.set_xlabel('예측 레이블', color='#7fa8c4', fontsize=11)
ax.set_ylabel('실제 레이블', color='#7fa8c4', fontsize=11)
ax.tick_params(colors='#7fa8c4')
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# CELL
# ─── 틀린 예측 이미지 수집 ───────────────────────────────────────────────
wrong_images, wrong_preds, wrong_labels = [], [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        mask = (predicted != labels.to(device))
        wrong_images.extend(images[mask].cpu())
        wrong_preds.extend(predicted[mask].cpu().numpy())
        wrong_labels.extend(labels[mask.cpu()].numpy())
        if len(wrong_images) >= 20:
            break

print(f'전체 오분류 샘플: {int((1-final_acc)*10000)}개 / 10,000개')

# 시각화
n_show = min(20, len(wrong_images))
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
fig.patch.set_facecolor('#0d1b2a')
fig.suptitle('❌ 틀린 예측 이미지 (빨강=예측, 초록=실제)', color='white', fontsize=13)

for i in range(n_show):
    ax = axes[i//10][i%10]
    ax.imshow(wrong_images[i].squeeze(), cmap='gray')
    ax.set_title(f'P:{wrong_preds[i]}\nT:{wrong_labels[i]}',
                 color='#ff5252', fontsize=9)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor('#ff5252')
        sp.set_linewidth(2)
        sp.set_visible(True)

plt.tight_layout()
plt.show()

# CELL
# ─── 테스트 이미지 1개 선택 ───────────────────────────────────────────────
sample_img, sample_label = test_dataset[0]
x = sample_img.unsqueeze(0).to(device)  # (1,1,28,28)

model.eval()
with torch.no_grad():
    feat1, feat2 = model.get_feature_maps(x)

print(f'입력 이미지 레이블: {sample_label}')
print(f'Conv Block 1 Feature Map: {feat1.shape}  → 32채널 × 14×14')
print(f'Conv Block 2 Feature Map: {feat2.shape}  → 64채널 × 7×7')

# ─── Conv Block 1 Feature Map (32개 필터) ────────────────────────────────
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor('#0d1b2a')
fig.suptitle(f'Conv Block 1 Feature Maps (숫자 "{sample_label}")  — 32 필터 × 14×14',
             color='white', fontsize=12)

# 원본 이미지
ax0 = fig.add_subplot(4, 9, 1)
ax0.imshow(sample_img.squeeze(), cmap='gray')
ax0.set_title('원본\n28×28', color='#00c9a7', fontsize=8)
ax0.axis('off')

# 각 필터의 특징 맵
for i in range(32):
    ax = fig.add_subplot(4, 9, i + 2)
    fm = feat1[0, i].cpu().numpy()
    ax.imshow(fm, cmap='viridis')
    ax.set_title(f'F{i+1}', color='#7fa8c4', fontsize=7)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ─── 활성화 크기별 Top-8 필터 ────────────────────────────────────────────
activations = feat1[0].cpu().numpy()
mean_act = activations.mean(axis=(1,2))
top8_idx = np.argsort(mean_act)[::-1][:8]

fig, axes = plt.subplots(2, 4, figsize=(13, 5))
fig.patch.set_facecolor('#0d1b2a')
fig.suptitle(f'가장 강하게 활성화된 Top-8 필터 (숫자 "{sample_label}")', color='white', fontsize=12)
for i, idx in enumerate(top8_idx):
    ax = axes[i//4][i%4]
    fm = activations[idx]
    ax.imshow(fm, cmap='plasma')
    ax.set_title(f'필터 #{idx+1}  (활성: {mean_act[idx]:.2f})', color='#00c9a7', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()

# CELL
# ✏️ 여기에 도전 과제 코드를 작성하세요!

# == batch_size ===
# 32 -> 
# 128 -> 

# 예시: Fashion-MNIST 로드
# fashion_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
# fashion_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
# ... (동일한 모델 구조로 학습)

print('도전 과제를 직접 구현해보세요! 🚀')

