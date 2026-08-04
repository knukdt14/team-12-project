"""
2단계 손상 종류 분류기 학습 스크립트 (dent/scratch/crack/glass shatter/lamp broken/tire flat)
1단계(YOLO26n)가 찾은 부위 박스를 크롭해 넣으면 손상 "종류"를 예측하는 경량 분류기.
데이터: damage_type_crops/{train,val,test}/<class>/*.jpg (build_damage_type_crops.py로 생성)
사용법: python train_damage_type.py
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR   = Path("damage_type_crops")
OUT_DIR    = Path("runs/damage_type_classifier")
IMG_SIZE   = 224   # 320으로 올렸다가 crop 변경과 맞물려 성능이 떨어져 원복 (DAMAGE_TYPE_CLASSIFIER.md 참고)
BATCH_SIZE = 32
EPOCHS     = 20
PATIENCE   = 5
LR         = 1e-3
FOCAL_GAMMA = 2.0  # focal loss 집중도 (클수록 어려운 샘플에 더 집중)
SEED       = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


class FocalLoss(nn.Module):
    """어려운(헷갈리는) 샘플에 더 집중하도록 CrossEntropy를 보정. crack/dent/scratch처럼
    클래스 간 유사도가 높은 경우 일반 CE보다 효과적 (DAMAGE_TYPE_CLASSIFIER.md 참고)"""

    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        # pt(모델 확신도)는 가중치 없는 순수 CE로 계산해야 함 — weight를 여기 섞으면
        # 가중치가 큰/작은 클래스의 pt가 왜곡되어 focal term이 이중으로 잘못 작용함
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce_loss
        if self.weight is not None:
            alpha_t = self.weight[targets]
            loss = alpha_t * loss
        return loss.mean()


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val", transform=EVAL_TF)
    class_names = train_ds.classes
    print(f"클래스: {class_names}")
    print(f"train: {len(train_ds)}장 / val: {len(val_ds)}장")

    # 클래스 불균형 보정 (tire flat/glass shatter 등 소수 클래스 가중치 부여)
    counts = Counter(label for _, label in train_ds.samples)
    weights = torch.tensor([1.0 / counts[i] for i in range(len(class_names))], dtype=torch.float32)
    weights = weights / weights.sum() * len(class_names)
    print("클래스 가중치:", {class_names[i]: round(w.item(), 3) for i, w in enumerate(weights)})

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model(len(class_names))
    criterion = FocalLoss(weight=weights.to(DEVICE), gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "class_names": class_names}, OUT_DIR / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"{PATIENCE}epoch 개선 없어 조기 종료")
                break

    print(f"\n=== 학습 완료, 최고 val acc: {best_val_acc:.4f} ===")
    print(f"모델 저장: {OUT_DIR / 'best.pt'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curve.png", dpi=150)
    print(f"학습 곡선 저장: {OUT_DIR / 'training_curve.png'}")


if __name__ == "__main__":
    main()
