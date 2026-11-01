"""Paper #8 (Afrin et al., ICCIT 2023, arXiv:2312.10701): the CNN and ResNet-50 character classifiers,
ported from the authors' Keras notebooks to PyTorch so they run on Kaggle (one GPU per job).

  python p08_train.py --task paper --model cnn      --data Data          --out runs/cnn_s42
  python p08_train.py --task paper --model resnet50 --data /tmp/p08_chars --out runs/resnet50 --max-hours 3
  python p08_train.py --fps-only --out runs/cnn_s42          # measure speed on an idle GPU

--task paper : a folder with training_data/, val_data/, testing_data/ and one sub-folder per class: the authors' own
               Data.zip (17 classes) or our token crops written by p08_chars.py (one folder per Bangla token)
--task team  : a split.csv (columns path,label,split,group); labels are the dataset's folders
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torchvision import models
from torchvision.transforms import v2 as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Settings taken from the paper (page 4-5) and the authors' notebooks (Code/CNN.ipynb, Code/resnet50.ipynb).
# imgsz: the paper text says 32x32, but the code trains the CNN on 28x28 and ResNet-50 on 150x150.
# lr: CNN uses Adam(1e-4); ResNet-50 is compiled with optimizer='adam', i.e. the Keras default 1e-3.
PAPER = {
    "cnn": dict(imgsz=28, epochs=40, batch=1, lr=1e-4),
    "resnet50": dict(imgsz=150, epochs=30, batch=1, lr=1e-3),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["paper", "team"])
    p.add_argument("--model", choices=["cnn", "resnet50"])
    p.add_argument("--data", help="paper: folder holding training_data/val_data/testing_data; team: split.csv")
    p.add_argument("--root", default="", help="team: folder that the paths in split.csv are relative to")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imagenet-norm", action="store_true", help="ResNet-50 only. The authors did NOT normalise (off = faithful).")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    p.add_argument("--max-hours", type=float, default=None, help="stop after the epoch that passes this cap")
    p.add_argument("--note", default="")
    p.add_argument("--fps-only", action="store_true", help="load <out>/best.pt and measure speed only")
    a = p.parse_args()
    if not a.fps_only:
        if not (a.task and a.model and a.data):
            p.error("--task, --model and --data are required for training")
        for k, v in PAPER[a.model].items():
            if getattr(a, k) is None:
                setattr(a, k, v)
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def list_paper_split(root):
    root = Path(root)
    dirs = {"train": "training_data", "val": "val_data", "test": "testing_data"}
    # Keras flow_from_directory sorts class folders the same way, so class ids match the authors' order.
    # Classes are taken from all three splits: a rare token can have test crops but no training crop (our data).
    classes = sorted({d.name for sub in dirs.values() if (root / sub).is_dir()
                      for d in (root / sub).iterdir() if d.is_dir()})
    items = {}
    for split, sub in dirs.items():
        items[split] = [(str(f), classes.index(c.name))
                        for c in sorted((root / sub).iterdir()) if c.is_dir()
                        for f in sorted(c.iterdir()) if f.suffix.lower() in IMG_EXT]
    return items, classes


def list_team_split(csv_path, root):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    classes = sorted({r["label"] for r in rows})
    root = Path(root) if root else Path(csv_path).parent
    items = {s: [(str(root / r["path"]), classes.index(r["label"])) for r in rows if r["split"] == s]
             for s in ("train", "val", "test")}
    return items, classes


def load_images(pairs, imgsz):
    """Decode and resize every image once, keep them in RAM as uint8 (N,3,H,W)."""
    x = torch.empty((len(pairs), 3, imgsz, imgsz), dtype=torch.uint8)
    for i, (path, _) in enumerate(pairs):
        im = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
        x[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    y = torch.tensor([lbl for _, lbl in pairs], dtype=torch.long)
    return x, y


def train_augment(model_name):
    if model_name == "cnn":   # ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
        return T.RandomAffine(degrees=0, translate=(0.1, 0.1))
    # ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True); Keras shear is in degrees
    return T.Compose([T.RandomAffine(degrees=0, shear=0.2, scale=(0.8, 1.2)), T.RandomHorizontalFlip()])


class TensorSet(torch.utils.data.Dataset):
    def __init__(self, x, y, aug=None):
        self.x, self.y, self.aug = x, y, aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.aug is not None:
            img = self.aug(img)
        return img.float() / 255.0, self.y[i]   # rescale=1./255, as in the paper's code


# ---------------------------------------------------------------- models
class PaperCNN(nn.Module):
    """Code/CNN.ipynb: 4 'same' convs (22,16,8,4 kernels) -> maxpool 4 -> 512 -> dropout 0.4 -> 128 -> classes."""

    def __init__(self, n_classes, imgsz):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 22, padding="same"), nn.ReLU(),
            nn.Conv2d(16, 32, 16, padding="same"), nn.ReLU(),
            nn.Conv2d(32, 64, 8, padding="same"), nn.ReLU(),
            nn.Conv2d(64, 128, 4, padding="same"), nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (imgsz // 4) ** 2, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        for m in self.modules():   # Keras defaults: glorot_uniform weights, zero bias
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))


class PaperResNet50(nn.Module):
    """Code/resnet50.ipynb: ImageNet ResNet-50 without top, all layers frozen, Flatten -> Dense(classes)."""

    def __init__(self, n_classes, imgsz, imagenet_norm=False, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        for p in self.backbone.parameters():
            p.requires_grad = False
        with torch.no_grad():
            feat = self.backbone.eval()(torch.zeros(1, 3, imgsz, imgsz)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(feat, n_classes))
        self.imagenet_norm = imagenet_norm
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # frozen Keras layers also run BatchNorm in inference mode
        return self

    def forward(self, x):
        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        with torch.no_grad():
            f = self.backbone(x)
        return self.fc(f)


def build_model(name, n_classes, imgsz, imagenet_norm=False, pretrained=True):
    if name == "cnn":
        return PaperCNN(n_classes, imgsz)
    return PaperResNet50(n_classes, imgsz, imagenet_norm, pretrained)


# ---------------------------------------------------------------- train / eval
def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu())
        gts.append(yb)
    return torch.cat(gts).numpy(), torch.cat(preds).numpy()


def metrics(y, p, classes):
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y, p),
        "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "report": classification_report(y, p, labels=list(range(len(classes))), target_names=classes, zero_division=0),
        "confusion": confusion_matrix(y, p, labels=list(range(len(classes)))).tolist(),
    }


@torch.no_grad()
def measure_fps(model, imgsz, device, n=300, warmup=50):
    """Model-only speed at batch 1 (the image is already resized and on the GPU)."""
    model.eval()
    x = torch.rand(1, 3, imgsz, imgsz, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return n / dt, 1000 * dt / n


def fps_only(a):
    out = Path(a.out)
    res = json.loads((out / "results.json").read_text())
    s = res["settings"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(s["model"], len(res["classes"]), s["imgsz"], s["imagenet_norm"], pretrained=False).to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    fps, ms = measure_fps(model, s["imgsz"], device)
    res["speed"] = {"fps_bs1": round(fps, 1), "ms_per_image": round(ms, 2), "device": gpu_name(device),
                    "note": "model only, batch 1, FP32, measured alone on an idle GPU"}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"{out.name}: {fps:.1f} img/s ({ms:.2f} ms) on {gpu_name(device)}")


def gpu_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU"


def main():
    a = get_args()
    if a.fps_only:
        return fps_only(a)
    seed_everything(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, classes = list_paper_split(a.data) if a.task == "paper" else list_team_split(a.data, a.root)
    if a.limit:
        items = {k: random.Random(a.seed).sample(v, min(a.limit, len(v))) for k, v in items.items()}
    print(f"[{out.name}] classes={len(classes)} train={len(items['train'])} val={len(items['val'])} "
          f"test={len(items['test'])} device={gpu_name(device)}", flush=True)

    t0 = time.time()
    data = {k: load_images(v, a.imgsz) for k, v in items.items()}
    print(f"[{out.name}] images loaded in {time.time() - t0:.0f}s", flush=True)
    aug = None if a.no_aug else train_augment(a.model)
    g = torch.Generator().manual_seed(a.seed)
    loaders = {
        "train": torch.utils.data.DataLoader(TensorSet(*data["train"], aug), batch_size=a.batch, shuffle=True, generator=g),
        "val": torch.utils.data.DataLoader(TensorSet(*data["val"]), batch_size=64),
        "test": torch.utils.data.DataLoader(TensorSet(*data["test"]), batch_size=64),
    }

    model = build_model(a.model, len(classes), a.imgsz, a.imagenet_norm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, eps=1e-7)

    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_sec"])
    best_acc, best_epoch, epochs_run = -1.0, 0, 0
    t_train = time.time()
    for ep in range(1, a.epochs + 1):
        te = time.time()
        tl, ta = run_epoch(model, loaders["train"], device, opt)
        vl, va = run_epoch(model, loaders["val"], device)
        sec = time.time() - te
        w.writerow([ep, f"{tl:.4f}", f"{ta:.4f}", f"{vl:.4f}", f"{va:.4f}", f"{sec:.1f}"])
        log.flush()
        if va > best_acc:   # keep the epoch with the best VALIDATION accuracy (never pick by test)
            best_acc, best_epoch = va, ep
            torch.save(model.state_dict(), out / "best.pt")
        print(f"[{out.name}] ep {ep:3d}/{a.epochs} loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f} "
              f"| {sec:.0f}s", flush=True)
        epochs_run = ep
        if a.max_hours and time.time() - t_train > a.max_hours * 3600 and ep < a.epochs:
            print(f"[time cap] {a.max_hours} h reached after epoch {ep}: stopping", flush=True)
            break
    train_min = (time.time() - t_train) / 60
    log.close()
    torch.save(model.state_dict(), out / "last.pt")

    y, p_last = predict(model, loaders["test"], device)          # last epoch, like the authors
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    y, p_best = predict(model, loaders["test"], device)          # best-validation checkpoint
    m_best, m_last = metrics(y, p_best, classes), metrics(y, p_last, classes)
    (out / "test_report.txt").write_text(m_best["report"])

    pc = lambda m: {"acc_%": round(100 * m["accuracy"], 2), "P_%": round(100 * m["precision_macro"], 2),
                    "R_%": round(100 * m["recall_macro"], 2), "F1_%": round(100 * m["f1_macro"], 2)}
    res = {
        "paper": "#8 Afrin et al. 2023, CNN and GFP-GAN (arXiv:2312.10701)",
        "run": out.name, "kind": "char-classifier", "note": a.note,
        "classes": classes, "n_classes": len(classes),
        "counts": {k: len(v) for k, v in items.items()},
        "settings": {"task": a.task, "model": a.model, "imgsz": a.imgsz, "epochs": a.epochs, "batch": a.batch,
                     "lr": a.lr, "optimizer": "Adam (eps 1e-7, Keras default)", "loss": "cross-entropy",
                     "augment": "none" if a.no_aug else str(aug), "imagenet_norm": a.imagenet_norm,
                     "resize": "PIL bilinear", "seed": a.seed, "data": a.data, "max_hours": a.max_hours},
        "params_total": n_params, "params_trainable": n_train,
        "epochs_planned": a.epochs, "epochs_run": epochs_run,
        "best_epoch": best_epoch, "val_acc_best": best_acc, "chars_val_best_%": round(100 * best_acc, 2),
        # in % for the results tables: 'last' = the model after the last epoch (what the paper reports),
        # 'best' = the checkpoint with the best validation accuracy (usual practice); P/R/F1 are macro averages
        "chars_test": {"last": pc(m_last), "best": pc(m_best)},
        "test_best": {k: v for k, v in m_best.items() if k != "report"},
        "test_last_epoch": {k: v for k, v in m_last.items() if k not in ("report", "confusion")},
        "train_minutes": round(train_min, 2),
        "train_device": gpu_name(device),
        "torch": torch.__version__,
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"[{out.name}] DONE best epoch {best_epoch} val {best_acc:.4f} | TEST acc {m_best['accuracy']:.4f} "
          f"macro-F1 {m_best['f1_macro']:.4f} (last epoch acc {m_last['accuracy']:.4f}) | {train_min:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
