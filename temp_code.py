# CELL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── 공통 색상 팔레트 ─────────────────────────────────────────────────────
COLORS = {
    'baseline':  '#7fa8c4',
    'momentum':  '#00c9a7',
    'adam':      '#00a8cc',
    'cosine':    '#a78bfa',
    'adamw':     '#fb7185',
    'swa':       '#ffb74d',
    'full':      '#69f0ae',
    'grad_clip': '#ff5252',
    'bg':        '#0d1b2a',
    'panel':     '#162236',
}

def set_dark_style(ax):
    """다크 테마 matplotlib 스타일 설정"""
    ax.set_facecolor(COLORS['panel'])
    ax.tick_params(colors='#7fa8c4')
    ax.xaxis.label.set_color('#7fa8c4')
    ax.yaxis.label.set_color('#7fa8c4')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e3a54')
    ax.grid(alpha=0.2, color='#1e3a54')

def plot_comparison(histories: dict, metric: str, title: str, ylabel: str, scale=100):
    """여러 실험 결과를 하나의 그래프에 비교"""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    set_dark_style(ax)
    for name, h in histories.items():
        vals = [v * scale for v in h[metric]]
        ax.plot(vals, label=name, linewidth=2, color=COLORS.get(name.lower().split()[0], '#ffffff'))
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(facecolor=COLORS['panel'], labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.show()

print('✅ 유틸리티 로드 완료')

# CELL
def make_landscape(landscape_type='nonconvex'):
    """간단한 2D 손실 지형 생성"""
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)

    if landscape_type == 'convex':
        Z = X**2 + Y**2  # 완벽한 그릇 모양
    else:
        # Non-convex: 여러 지역 최솟값 + 안장점
        Z = (np.sin(X*1.5) + np.cos(Y*1.5)           # 진동
             + 0.3 * (X**2 + Y**2)                   # 전체적 기울기
             + 0.5 * np.sin(X*Y)                     # 교차항
             - 0.5 * np.exp(-(X**2 + Y**2) / 2))     # 중앙 함정
    return X, Y, Z


fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d'})
fig.patch.set_facecolor(COLORS['bg'])

for ax, ltype, title in zip(axes, ['convex','nonconvex'],
                             ['Convex Loss Landscape\n(선형 회귀, 로지스틱)', 'Non-Convex Loss Landscape\n(딥러닝 CNN)']):
    X, Y, Z = make_landscape(ltype)
    ax.set_facecolor('#0d1b2a')
    cmap = 'YlGnBu' if ltype == 'convex' else 'plasma'
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, rstride=5, cstride=5)
    ax.set_title(title, color='white', fontsize=11, pad=10)
    ax.tick_params(colors='#7fa8c4', labelsize=7)
    ax.set_xlabel('θ₁', color='#7fa8c4', fontsize=9)
    ax.set_ylabel('θ₂', color='#7fa8c4', fontsize=9)
    ax.set_zlabel('Loss', color='#7fa8c4', fontsize=9)

plt.suptitle('Loss Landscape 비교', color='white', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# 2D 등고선 + Gradient 방향
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(COLORS['bg'])
for ax, ltype, title in zip(axes, ['convex','nonconvex'],
                             ['Convex (SGD 경로)', 'Non-Convex (SGD 경로 — 함정 존재)']):
    X, Y, Z = make_landscape(ltype)
    set_dark_style(ax)
    cs = ax.contourf(X, Y, Z, levels=25, cmap='plasma', alpha=0.8)
    fig.colorbar(cs, ax=ax, label='Loss')

    # SGD 경로 시뮬레이션
    pos = np.array([2.5, 2.5])
    path = [pos.copy()]
    lr = 0.05
    for _ in range(80):
        xi, yi = pos
        if ltype == 'convex':
            grad = np.array([2*xi, 2*yi])
        else:
            gx = 1.5*np.cos(xi*1.5) + 0.6*xi + 0.5*yi*np.cos(xi*yi) + xi*np.exp(-(xi**2+yi**2)/2)
            gy = -1.5*np.sin(yi*1.5) + 0.6*yi + 0.5*xi*np.cos(xi*yi) + yi*np.exp(-(xi**2+yi**2)/2)
            grad = np.array([gx, gy])
        pos = pos - lr * grad
        pos = np.clip(pos, -3, 3)
        path.append(pos.copy())
    path = np.array(path)
    ax.plot(path[:,0], path[:,1], 'w-', alpha=0.8, linewidth=2, label='SGD 경로')
    ax.plot(path[0,0], path[0,1], 'go', markersize=9, label='시작점')
    ax.plot(path[-1,0], path[-1,1], 'r*', markersize=12, label='수렴점')
    ax.set_title(title, color='white', fontsize=11)
    ax.legend(facecolor=COLORS['panel'], labelcolor='white', fontsize=9)

plt.suptitle('SGD 최적화 경로: Convex vs Non-Convex', color='white', fontsize=13)
plt.tight_layout()
plt.show()

# CELL
# ─── 7-Segment 데이터 (공통) ──────────────────────────────────────────────
SEGMENTS = {
    0:[1,1,1,1,1,1,0], 1:[0,1,1,0,0,0,0], 2:[1,1,0,1,1,0,1],
    3:[1,1,1,1,0,0,1], 4:[0,1,1,0,0,1,1], 5:[1,0,1,1,0,1,1],
    6:[1,0,1,1,1,1,1], 7:[1,1,1,0,0,0,0], 8:[1,1,1,1,1,1,1],
    9:[1,1,1,1,0,1,1],
}

def make_seg_data(noise_prob=0.05, samples=200):
    X, y = [], []
    for digit, pat in SEGMENTS.items():
        for _ in range(samples):
            noise = np.random.binomial(1, noise_prob, 7)
            X.append((np.array(pat) ^ noise).astype(np.float32))
            y.append(digit)
    X, y = np.array(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return (torch.FloatTensor(Xtr).to(device), torch.LongTensor(ytr).to(device),
            torch.FloatTensor(Xte).to(device), torch.LongTensor(yte).to(device))


class SevenSegNet(nn.Module):
    def __init__(self, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 10),
        )
    def forward(self, x): return self.net(x)


def train_seg(opt_fn, sched_fn=None, epochs=120, noise=0.05, grad_clip=None, label_smooth=0.0):
    """7-Segment 학습 실험 함수."""
    Xtr, ytr, Xte, yte = make_seg_data(noise_prob=noise)
    model = SevenSegNet(64, 32).to(device)
    optimizer = opt_fn(model.parameters())
    scheduler = sched_fn(optimizer) if sched_fn else None
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    history = {'train_loss':[], 'train_acc':[], 'test_acc':[], 'lr':[]}

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(Xtr)
        loss = criterion(out, ytr)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler: scheduler.step()

        _, pred_tr = torch.max(out, 1)
        tr_acc = (pred_tr == ytr).float().mean().item()

        model.eval()
        with torch.no_grad():
            _, pred_te = torch.max(model(Xte), 1)
            te_acc = (pred_te == yte).float().mean().item()

        history['train_loss'].append(loss.item())
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

    return history, model


X_tr, y_tr, X_te, y_te = make_seg_data(0.05)
print(f'7-Segment 데이터 준비 완료: Train={len(y_tr)}, Test={len(y_te)}')

# CELL
# ─── 실험 1: 옵티마이저 비교 (7-Segment, noise=5%) ────────────────────────
print('실험 실행 중... (약 30초 소요)')

results_seg = {}

# 베이스라인: Adam
results_seg['Baseline (Adam)'], _ = train_seg(
    opt_fn=lambda p: optim.Adam(p, lr=0.001), epochs=120)
print('✅ Baseline 완료')

# Momentum SGD + Nesterov
results_seg['Momentum (SGD+NAG)'], _ = train_seg(
    opt_fn=lambda p: optim.SGD(p, lr=0.01, momentum=0.9, nesterov=True), epochs=120)
print('✅ Momentum 완료')

# Adam + StepLR
results_seg['Adam + StepLR'], _ = train_seg(
    opt_fn=lambda p: optim.Adam(p, lr=0.001),
    sched_fn=lambda opt: optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5),
    epochs=120)
print('✅ Adam+StepLR 완료')

# Adam + CosineAnnealingWarmRestarts
results_seg['Adam + CosineWarmRestart'], _ = train_seg(
    opt_fn=lambda p: optim.Adam(p, lr=0.001),
    sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2),
    epochs=120)
print('✅ CosineWarmRestart 완료')

# AdamW
results_seg['AdamW'], _ = train_seg(
    opt_fn=lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4), epochs=120)
print('✅ AdamW 완료')

# CELL
# ─── 시각화: 7-Segment 옵티마이저 비교 ───────────────────────────────────
color_map = {
    'Baseline (Adam)':          COLORS['baseline'],
    'Momentum (SGD+NAG)':       COLORS['momentum'],
    'Adam + StepLR':            COLORS['cosine'],
    'Adam + CosineWarmRestart': COLORS['adam'],
    'AdamW':                    COLORS['adamw'],
}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('7-Segment: Optimizer & Scheduler 비교 (noise=5%)', color='white', fontsize=13)

metrics = [('train_loss', 'Train Loss', 1), ('test_acc', 'Test Accuracy (%)', 100), ('lr', 'Learning Rate', 1)]
for ax, (key, ylabel, scale) in zip(axes, metrics):
    set_dark_style(ax)
    for name, h in results_seg.items():
        ax.plot([v*scale for v in h[key]], label=name, lw=2, color=color_map[name])
    ax.set_title(ylabel, color='white')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if key == 'test_acc':
        ax.axhline(100, color='#69f0ae', lw=1, linestyle=':', alpha=0.7)
        ax.set_ylim(80, 105)
    ax.legend(facecolor=COLORS['panel'], labelcolor='white', fontsize=8)

plt.tight_layout()
plt.show()

# 최종 결과 요약
print('\n최종 Test Accuracy 비교:')
print(f'{"실험":>30} | {"최종 Test Acc":>14} | {"수렴 Epoch":>10}')
print('-' * 60)
for name, h in results_seg.items():
    final_acc = h['test_acc'][-1] * 100
    converge = next((i for i,a in enumerate(h['test_acc']) if a >= 0.99), len(h['test_acc']))
    print(f'{name:>30} | {final_acc:>13.2f}% | {converge:>9}ep')

# CELL
# ─── 고노이즈 환경 실험 (noise=15%) ─────────────────────────────────────
print('고노이즈(15%) 실험 실행 중...')

results_noisy = {}

# 베이스라인
results_noisy['Baseline (Adam)'], _ = train_seg(
    opt_fn=lambda p: optim.Adam(p, lr=0.001), epochs=200, noise=0.15)
print('✅ Baseline 완료')

# Gradient Clipping 추가
results_noisy['Adam + GradClip(1.0)'], _ = train_seg(
    opt_fn=lambda p: optim.Adam(p, lr=0.001),
    epochs=200, noise=0.15, grad_clip=1.0)
print('✅ GradClip 완료')

# CosineWarmRestart + Grad Clip
results_noisy['Cosine+GradClip'], _ = train_seg(
    opt_fn=lambda p: optim.Adam(p, lr=0.001),
    sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2),
    epochs=200, noise=0.15, grad_clip=1.0)
print('✅ Cosine+GradClip 완료')

# AdamW + CosineWarmRestart + Grad Clip (Full Recipe)
results_noisy['Full Recipe (AdamW+Cosine+Clip)'], _ = train_seg(
    opt_fn=lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
    sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2),
    epochs=200, noise=0.15, grad_clip=1.0)
print('✅ Full Recipe 완료')

# ─── 시각화 ─────────────────────────────────────────────────────────────
color_map2 = {
    'Baseline (Adam)':            COLORS['baseline'],
    'Adam + GradClip(1.0)':       COLORS['grad_clip'],
    'Cosine+GradClip':            COLORS['cosine'],
    'Full Recipe (AdamW+Cosine+Clip)': COLORS['full'],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('7-Segment 고노이즈(15%) 환경 — Gradient Clipping & Warm Restart 효과', color='white', fontsize=12)

for ax, key, ylabel, scale in [
    (axes[0], 'train_loss', 'Train Loss', 1),
    (axes[1], 'test_acc',   'Test Accuracy (%)', 100)
]:
    set_dark_style(ax)
    for name, h in results_noisy.items():
        ax.plot([v*scale for v in h[key]], label=name, lw=2, color=color_map2[name])
    ax.set_title(ylabel, color='white')
    ax.set_xlabel('Epoch')
    ax.legend(facecolor=COLORS['panel'], labelcolor='white', fontsize=8.5)

plt.tight_layout()
plt.show()

# ─── Gradient 노름 변화 시각화 ────────────────────────────────────────────
print('\nGradient Norm 변화 측정...')

def measure_grad_norm(noise=0.15, epochs=100, use_clip=False):
    """학습 중 그래디언트 노름을 기록"""
    Xtr, ytr, Xte, yte = make_seg_data(noise_prob=noise, samples=200)
    model = SevenSegNet(128, 64).to(device)  # 더 큰 모델로 폭발 확인
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    grad_norms = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        if use_clip:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return grad_norms

norms_no_clip   = measure_grad_norm(noise=0.15, use_clip=False)
norms_with_clip = measure_grad_norm(noise=0.15, use_clip=True)

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor(COLORS['bg'])
set_dark_style(ax)
ax.plot(norms_no_clip,   color=COLORS['grad_clip'], lw=2, label='Gradient Norm (클리핑 없음)', alpha=0.85)
ax.plot(norms_with_clip, color=COLORS['full'],      lw=2, label='Gradient Norm (클리핑 max=1.0)')
ax.axhline(1.0, color=COLORS['amber'], lw=1.5, linestyle='--', alpha=0.7, label='클리핑 임계값 (τ=1.0)')
ax.set_title('Gradient Norm 변화: Clipping 전/후', color='white', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient Norm (L2)')
ax.legend(facecolor=COLORS['panel'], labelcolor='white')
plt.tight_layout()
plt.show()

# CELL
# ─── MNIST 데이터 & 모델 ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)


class MnistCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128,10)
        )
    def forward(self, x): return self.classifier(self.features(x))


def train_mnist_epoch(model, loader, criterion, optimizer, grad_clip=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss/total, correct/total


@torch.no_grad()
def eval_mnist(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss/total, correct/total

print('✅ MNIST 데이터 및 모델 준비 완료')

# CELL
# ─── 실험 A: 베이스라인 (Adam + CrossEntropy) ────────────────────────────
EPOCHS = 15
results_mnist = {}

def run_mnist_exp(name, opt_fn, sched_fn=None, smooth=0.0, grad_clip=None, use_swa=False, swa_start=10):
    model = MnistCNN(dropout=0.5).to(device)
    optimizer = opt_fn(model.parameters())
    scheduler = sched_fn(optimizer) if sched_fn else None
    criterion = nn.CrossEntropyLoss(label_smoothing=smooth)
    h = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'lr':[]}

    swa_model = AveragedModel(model).to(device) if use_swa else None
    swa_sched  = SWALR(optimizer, swa_lr=0.005) if use_swa else None

    for ep in range(1, EPOCHS+1):
        tl, ta = train_mnist_epoch(model, train_loader, criterion, optimizer, grad_clip)
        vl, va = eval_mnist(model, test_loader, criterion)

        if use_swa and ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        elif scheduler:
            scheduler.step()

        h['train_loss'].append(tl); h['train_acc'].append(ta)
        h['test_loss'].append(vl);  h['test_acc'].append(va)
        h['lr'].append(optimizer.param_groups[0]['lr'])
        print(f'  [{name}] Ep {ep:>2}/{EPOCHS}  Train:{ta*100:.2f}%  Test:{va*100:.2f}%')

    # SWA BN 업데이트
    if use_swa:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        _, swa_acc = eval_mnist(swa_model, test_loader, criterion)
        print(f'  >>> SWA 최종 Test Acc: {swa_acc*100:.3f}%')
        h['swa_final_acc'] = swa_acc

    results_mnist[name] = h
    return h


print('=== 실험 A: Baseline (Adam) ===')
run_mnist_exp('Baseline',
              opt_fn=lambda p: optim.Adam(p, lr=0.001))

print('\n=== 실험 B: Adam + CosineAnnealing ===')
run_mnist_exp('Adam+Cosine',
              opt_fn=lambda p: optim.Adam(p, lr=0.001),
              sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5))

# CELL
print('=== 실험 C: AdamW + LabelSmoothing ===')
run_mnist_exp('AdamW+LabelSmooth',
              opt_fn=lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
              sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5),
              smooth=0.1)

print('\n=== 실험 D: Full Recipe (AdamW + Cosine + LabelSmooth + SWA) ===')
run_mnist_exp('Full Recipe (SWA)',
              opt_fn=lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
              sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5),
              smooth=0.1, use_swa=True, swa_start=8)

# CELL
# ─── MNIST 실험 결과 시각화 ───────────────────────────────────────────────
color_map_mnist = {
    'Baseline':            COLORS['baseline'],
    'Adam+Cosine':         COLORS['cosine'],
    'AdamW+LabelSmooth':   COLORS['adamw'],
    'Full Recipe (SWA)':   COLORS['full'],
}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('MNIST CNN: Non-Convex 기법 조합 비교', color='white', fontsize=13)

for ax, key, ylabel, scale in [
    (axes[0], 'train_loss', 'Train Loss', 1),
    (axes[1], 'test_acc',   'Test Accuracy (%)', 100),
    (axes[2], 'lr',         'Learning Rate', 1),
]:
    set_dark_style(ax)
    for name, h in results_mnist.items():
        c = color_map_mnist.get(name, '#ffffff')
        lw = 2.5 if 'Full' in name else 1.8
        ax.plot([v*scale for v in h[key]], label=name, lw=lw, color=c)
    ax.set_title(ylabel, color='white')
    ax.set_xlabel('Epoch')
    if key == 'test_acc':
        ax.axhline(99, color='#ffb74d', lw=1, linestyle=':', alpha=0.8, label='99% 목표')
        ax.set_ylim(97.5, 100.5)
    ax.legend(facecolor=COLORS['panel'], labelcolor='white', fontsize=8.5)

plt.tight_layout()
plt.show()

# 최종 결과 요약
print('\n최종 Test Accuracy 비교 (MNIST):')
print(f'{"실험":>30} | {"최종 Test Acc":>14} | {"SWA Acc":>10}')
print('-' * 60)
for name, h in results_mnist.items():
    final_acc = h['test_acc'][-1]*100
    swa_acc_str = f"{h.get('swa_final_acc', 0)*100:.3f}%" if 'swa_final_acc' in h else '—'
    print(f'{name:>30} | {final_acc:>13.3f}% | {swa_acc_str:>10}')

# CELL
# ─── 전체 결과 비교 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('전체 실험 최종 Test Accuracy 비교', color='white', fontsize=14)

# 7-Segment 막대 그래프
ax1 = axes[0]
set_dark_style(ax1)
seg_names = list(results_seg.keys())
seg_accs  = [results_seg[n]['test_acc'][-1]*100 for n in seg_names]
bars1 = ax1.barh(seg_names, seg_accs,
                 color=[COLORS.get(n.lower().split()[0], '#7fa8c4') for n in seg_names],
                 alpha=0.85, height=0.6)
for bar, acc in zip(bars1, seg_accs):
    ax1.text(bar.get_width()-0.5, bar.get_y()+bar.get_height()/2,
             f'{acc:.2f}%', va='center', ha='right', color='white', fontweight='bold', fontsize=9)
ax1.set_title('7-Segment (noise=5%)', color='white', fontsize=12)
ax1.set_xlabel('Test Accuracy (%)')
ax1.set_xlim(90, 102)
ax1.axvline(100, color='#69f0ae', lw=1.5, linestyle='--', alpha=0.7)

# MNIST 막대 그래프
ax2 = axes[1]
set_dark_style(ax2)
mnist_names = list(results_mnist.keys())
mnist_accs  = [results_mnist[n].get('swa_final_acc', results_mnist[n]['test_acc'][-1])*100 for n in mnist_names]
bar_colors  = [COLORS['baseline'], COLORS['cosine'], COLORS['adamw'], COLORS['full']]
bars2 = ax2.barh(mnist_names, mnist_accs, color=bar_colors, alpha=0.85, height=0.5)
for bar, acc in zip(bars2, mnist_accs):
    ax2.text(bar.get_width()-0.01, bar.get_y()+bar.get_height()/2,
             f'{acc:.3f}%', va='center', ha='right', color='white', fontweight='bold', fontsize=9)
ax2.set_title('MNIST CNN', color='white', fontsize=12)
ax2.set_xlabel('Test Accuracy (%)')
ax2.set_xlim(97.5, 100.5)
ax2.axvline(99, color='#ffb74d', lw=1.5, linestyle='--', alpha=0.7, label='99% 목표')
ax2.legend(facecolor=COLORS['panel'], labelcolor='white', fontsize=9)

plt.tight_layout()
plt.show()

print('\n결론:')
print('  7-Segment: 데이터가 단순하여 대부분 100% 달성. 고노이즈 환경에서 기법 효과 명확.')
print('  MNIST: Label Smoothing + SWA 조합이 베이스라인 대비 정확도 및 일반화 향상.')
print('         CosineAnnealing은 후반 미세조정을 도와 수렴 품질 개선.')

# CELL
# ✏️ 도전 과제 코드 공간

# ── 예시: 노이즈 수준별 최적 기법 탐색 ──────────────────────────────────
# noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
# best_methods = {}
# for noise in noise_levels:
#     h_base,  _ = train_seg(lambda p: optim.Adam(p, lr=0.001), epochs=150, noise=noise)
#     h_cosine,_ = train_seg(
#         lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
#         sched_fn=lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40),
#         epochs=150, noise=noise, grad_clip=1.0)
#     diff = h_cosine['test_acc'][-1] - h_base['test_acc'][-1]
#     print(f'noise={noise:.2f}: baseline={h_base["test_acc"][-1]*100:.1f}%,'
#           f' cosine+clip={h_cosine["test_acc"][-1]*100:.1f}%, 개선={diff*100:.2f}%p')

print('도전 과제를 직접 구현해보세요! 🚀')

