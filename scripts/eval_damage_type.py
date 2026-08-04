"""
손상 종류 분류기(runs/damage_type_classifier/best.pt) test셋 최종 평가.
클래스별 precision/recall/f1 + confusion matrix 출력.
사용법: python eval_damage_type.py
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR   = Path("damage_type_crops")
CKPT_PATH  = Path("runs/damage_type_classifier/best.pt")
IMG_SIZE   = 224  # train_damage_type.py와 동일하게 맞춤
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    class_names = ckpt["class_names"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=EVAL_TF)
    assert test_ds.classes == class_names, f"클래스 순서 불일치: {test_ds.classes} vs {class_names}"
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    n = len(class_names)
    confusion = [[0] * n for _ in range(n)]
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu()
            for t, p in zip(labels, preds):
                confusion[t.item()][p.item()] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"=== test 종합 정확도: {correct}/{total} = {correct/total:.4f} ===\n")

    print(f"{'클래스':<15}{'Precision':>10}{'Recall':>10}{'F1':>10}{'개수':>8}")
    for i, name in enumerate(class_names):
        tp = confusion[i][i]
        fn = sum(confusion[i]) - tp
        fp = sum(confusion[r][i] for r in range(n)) - tp
        support = sum(confusion[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{name:<15}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{support:>8}")

    print("\n=== Confusion Matrix (행=실제, 열=예측) ===")
    header = "".join(f"{n[:6]:>8}" for n in class_names)
    print(f"{'':<15}{header}")
    for i, name in enumerate(class_names):
        row = "".join(f"{confusion[i][j]:>8}" for j in range(n))
        print(f"{name:<15}{row}")


if __name__ == "__main__":
    main()
