import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from geometry.layers import YggdrasilSandwichModel

def run_benchmark():
    print("================================================================")
    print("  TRAINING YGGDRASIL GEOMETRIC SANDWICH ON NVIDIA RTX 3060")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load real handwritten digits (1797 samples, 64 features, 10 classes)
    digits = load_digits()
    X = digits.data.astype("float32") / 16.0  # normalize [0, 1]
    y = digits.target.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    print(f"Dataset: Handwritten Digits (Train: {len(X_train)}, Test: {len(X_test)})")

    # 1. Train Yggdrasil Geometric Sandwich Model
    dim_h = 16
    dim_t = 16
    num_prototypes = 20
    hidden_dim = 64

    model = YggdrasilSandwichModel(
        in_features=64,
        hidden_dim=hidden_dim,
        num_prototypes=num_prototypes,
        num_classes=10,
        dim_h=dim_h,
        dim_t=dim_t,
        c=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 40
    batch_size = 128
    num_batches = (len(X_train) + batch_size - 1) // batch_size

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        total_loss = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            bx, by = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()

            optimizer.step()
            model.cortex.clamp_prototypes()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t).argmax(dim=-1)
                test_acc = (test_preds == y_test_t).float().mean().item() * 100.0
                print(f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss / num_batches:.4f} | Test Acc: {test_acc:.2f}%")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print(f"\nTraining Completed in: {elapsed:.2f} seconds ({epochs * len(X_train) / elapsed:.1f} samples/sec)")

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).argmax(dim=-1)
        final_acc = (final_preds == y_test_t).float().mean().item() * 100.0
    print(f"FINAL TEST ACCURACY ON RTX 3060: {final_acc:.2f}%\n")
    assert final_acc >= 97.0, f"Expected >= 97% accuracy, got {final_acc:.2f}%"

if __name__ == "__main__":
    run_benchmark()
