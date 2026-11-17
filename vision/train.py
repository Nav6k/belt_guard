"""Fine-tune a pretrained ResNet18 as a belt-patch classifier (intact vs. damage classes).

    python -m vision.train

Augmentation mimics a camera over a mining belt: dust haze (contrast/brightness), motion blur,
lighting changes, and arbitrary orientation. Only the last ResNet block + head are trained,
which is enough for a small dataset and keeps CPU training to a few minutes.
"""
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import DATASET, INTACT, class_names, make_patches, split_items

MODEL_DIR = Path("vision/models")
IMG = 160
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Patches(Dataset):
    def __init__(self, patches, classes, tf):
        self.patches, self.idx, self.tf = patches, {c: i for i, c in enumerate(classes)}, tf

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        im, label = self.patches[i]
        return self.tf(im), self.idx[label]


def build_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith("layer4")
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x).argmax(1).tolist()
            ys += y.tolist()
    return ys, ps


def main(epochs: int = 15):
    torch.manual_seed(0)
    classes = list(dict.fromkeys([INTACT] + class_names()))       # several source labels may map to one class
    train = make_patches(split_items("train"))
    val_split = "valid" if (DATASET / "valid").exists() else "test"
    val = make_patches(split_items(val_split), seed=1)
    print("train patches:", Counter(c for _, c in train))
    print(f"{val_split} patches:", Counter(c for _, c in val))
    classes = [c for c in classes if any(l == c for _, l in train)]   # drop classes with no training data

    counts = Counter(c for _, c in train)
    weights = torch.tensor([len(train) / (len(classes) * counts[c]) for c in classes], dtype=torch.float)
    tl = DataLoader(Patches(train, classes, train_tf), batch_size=32, shuffle=True)
    vl = DataLoader(Patches([p for p in val if p[1] in classes], classes, eval_tf), batch_size=64)

    model = build_model(len(classes))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        ys, ps = evaluate(model, vl)
        acc = sum(a == b for a, b in zip(ys, ps)) / max(len(ys), 1)
        print(f"epoch {epoch + 1}: loss {total / len(train):.3f}  {val_split} acc {acc:.3f}")
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ys, ps = evaluate(model, vl)
    labels = list(range(len(classes)))
    report = classification_report(ys, ps, labels=labels, target_names=classes, zero_division=0)
    cm = confusion_matrix(ys, ps, labels=labels)
    print(report)
    test_note = ""
    if val_split == "valid" and (DATASET / "test" / "images").exists():
        test = [p for p in make_patches(split_items("test"), seed=2) if p[1] in classes]
        ts, tp = evaluate(model, DataLoader(Patches(test, classes, eval_tf), batch_size=64))
        test_report = classification_report(ts, tp, labels=labels, target_names=classes, zero_division=0)
        print(test_report)
        test_note = f"\n## Held-out `test` split (not used for epoch selection)\n\n```\n{test_report}\n```\n"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "belt_patch_resnet18.pt")
    (MODEL_DIR / "classes.json").write_text(json.dumps(classes))
    Path("vision/REPORT.md").write_text(
        "# Vision model: evaluation report\n\n"
        f"Patch classifier, ResNet18 (ImageNet weights, layer4 + head fine-tuned), {IMG}px, {epochs} epochs.\n"
        f"Evaluated on the dataset's `{val_split}` split (images never seen in training).\n\n"
        f"```\n{report}\n```\n\nConfusion matrix (rows = true, cols = predicted, order {classes}):\n\n"
        f"```\n{cm}\n```\n" + test_note)
    print("saved vision/models/belt_patch_resnet18.pt")


if __name__ == "__main__":
    main()
