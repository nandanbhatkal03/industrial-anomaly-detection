from __future__ import annotations

import time

import torch

from src.utils.logger import get_logger

log = get_logger()


def train_model(model, method: str, cfg: dict, train_loader, val_loader, device) -> dict:
    """Fit any model. One-pass models (PaDiM/PatchCore/LightPatch) call model.fit();
    gradient-trained ones (autoencoder, student-teacher) run an Adam loop using cfg['training'][method]."""
    t0 = time.perf_counter()
    model.to(device)
    history = []
    if not model.needs_gradient_training:
        model.fit(train_loader, device, log=log.info)
    else:
        tc = cfg.get("training", {}).get(method, {})
        epochs, lr, wd = int(tc.get("epochs", 50)), float(tc.get("lr", 1e-3)), float(tc.get("weight_decay", 0.0))
        opt = torch.optim.Adam(model.trainable_parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        for ep in range(1, epochs + 1):
            model.train()
            total, n = 0.0, 0
            for batch in train_loader:
                x = batch["image"].to(device)
                loss = model.loss(x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * x.shape[0]
                n += x.shape[0]
            sched.step()
            val = None
            if val_loader is not None and (ep % 5 == 0 or ep == epochs):
                model.eval()
                with torch.no_grad():
                    vs = [model.loss(b["image"].to(device)).item() * b["image"].shape[0] for b in val_loader]
                val = sum(vs) / max(len(val_loader.dataset), 1)
            history.append({"epoch": ep, "train_loss": total / max(n, 1), "val_loss": val})
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                log.info(f"[{method}] epoch {ep}/{epochs} train_loss={total / max(n, 1):.5f}"
                         + (f" val_loss={val:.5f}" if val is not None else ""))
    model.eval()
    return {"train_time_s": time.perf_counter() - t0, "history": history}
