import os
import json
import math
import time
import glob
import torch
from torch.utils.data import DataLoader
from typing import List, Optional


class Trainer:

    def __init__(self, model, tokenizer, train_dl, val_dl, cfg, device):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.train_dl  = train_dl
        self.val_dl    = val_dl
        self.cfg       = cfg
        self.device    = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg.training.lr,
            betas        = (0.9, 0.95),
            weight_decay = 0.1,
        )

        self.scheduler     = self._build_scheduler()
        self.best_val_loss = float("inf")
        self.history       = {"epochs": []}

        print(f"  🖥️  Device     : {device}")
        print(f"  🧠 Parâmetros : {model.num_params():,}")
        print(f"  📈 LR         : {cfg.training.lr}")

    def _build_scheduler(self):
        sc    = self.cfg.training.scheduler
        total = self.cfg.training.epochs * len(self.train_dl)

        if not sc.enabled or sc.type == "none":
            return None

        warmup = sc.warmup_epochs * len(self.train_dl)

        if sc.type == "cosine":
            from torch.optim.lr_scheduler import (
                LinearLR, CosineAnnealingLR, SequentialLR,
            )
            warm   = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup
            )
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=total - warmup
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warm, cosine],
                milestones=[warmup],
            )

        if sc.type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total,
            )

        return None

    # ── Treino ──────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total    = 0.0
        t0       = time.time()
        n        = len(self.train_dl)
        log_every = max(1, n * self.cfg.logging.log_every_pct // 100)

        for step, (x, y) in enumerate(self.train_dl):
            x, y     = x.to(self.device), y.to(self.device)
            _, loss  = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.grad_clip,
            )
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total += loss.item()

            if (step + 1) % log_every == 0:
                pct = 100 * (step + 1) / n
                print(
                    f"    Epoch {epoch} │ {pct:5.1f}% │ "
                    f"loss={loss.item():.4f} │ "
                    f"{time.time() - t0:.1f}s"
                )

        return total / n

    # ── Avaliação ────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total = 0.0
        for x, y in self.val_dl:
            x, y    = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total  += loss.item()
        return total / max(len(self.val_dl), 1)

    # ── Amostra ──────────────────────────────────────────

    @torch.no_grad()
    def _sample(self, prompt: str) -> str:
        self.model.eval()
        ids   = self.tokenizer.encode(prompt)
        ids_t = torch.tensor([ids], device=self.device)
        out   = self.model.generate(
            ids_t,
            max_new_tokens = self.cfg.generation.max_new_tokens,
            temperature    = self.cfg.generation.temperature,
            top_k          = self.cfg.generation.top_k,
            eos_id         = self.tokenizer.eos_id,
        )
        new_ids = out[0, len(ids):].tolist()
        return self.tokenizer.decode(new_ids, skip_special=True)

    # ── Checkpoint ───────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float):
        ck_dir = self.cfg.paths.checkpoints
        fname  = os.path.join(ck_dir, f"epoch_{epoch:03d}.pt")

        torch.save({
            "epoch":    epoch,
            "val_loss": val_loss,
            "cfg":      self.model.cfg,
            "state":    self.model.state_dict(),
        }, fname)

        # Manter apenas os N mais recentes
        keep   = self.cfg.checkpoint.keep_last_n
        all_ck = sorted(glob.glob(os.path.join(ck_dir, "epoch_*.pt")))
        for old in all_ck[:-keep]:
            os.remove(old)

    def _save_best(self):
        self.model.save(self.cfg.paths.latest_model)

    def _save_history(self):
        path = os.path.join(self.cfg.paths.logs, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # ── Ciclo principal ───────────────────────────────────

    def fit(self, epochs: int, test_prompts: Optional[List[str]] = None):
        prompts = (test_prompts or [])[:self.cfg.logging.num_samples]

        print(f"\n{'═' * 55}")
        print(f"  🚀 Ciclo Treino → Teste → Repita")
        print(f"  📅 {epochs} epochs")
        print(f"{'═' * 55}\n")

        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── TREINO ──────────────────────────────────
            print(f"  ┌─ EPOCH {epoch}/{epochs} ─ TREINO ─────────────")
            train_loss = self._train_epoch(epoch)
            print(
                f"  │  Train Loss : {train_loss:.4f} │ "
                f"PPL : {math.exp(train_loss):.2f}"
            )

            # ── AVALIAÇÃO ────────────────────────────────
            print(f"  ├─ EPOCH {epoch}/{epochs} ─ AVALIAÇÃO ──────────")
            val_loss = self._evaluate()
            ppl      = math.exp(val_loss)
            improved = val_loss < self.best_val_loss

            print(
                f"  │  Val Loss   : {val_loss:.4f} │ "
                f"PPL : {ppl:.2f} "
                f"{'⭐ Melhor!' if improved else ''}"
            )

            # ── CHECKPOINT ───────────────────────────────
            if improved:
                self.best_val_loss = val_loss
                self._save_best()
                print(f"  │  💾 Melhor modelo → {self.cfg.paths.latest_model}")

            if epoch % self.cfg.checkpoint.save_every_n == 0:
                self._save_checkpoint(epoch, val_loss)
                print(f"  │  💾 Checkpoint epoch {epoch} salvo")

            # ── AMOSTRAS ─────────────────────────────────
            if self.cfg.logging.show_samples and prompts:
                print(f"  ├─ EPOCH {epoch}/{epochs} ─ AMOSTRAS ─────────")
                for prompt in prompts:
                    resp    = self._sample(prompt)
                    display = prompt.replace("<usr>", "").replace("<sep>", "")
                    print(f"  │  💬 {display}")
                    print(f"  │  🤖 {resp[:100]}")
                    print(f"  │")

            # ── HISTÓRICO ────────────────────────────────
            self.history["epochs"].append({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "ppl":        round(ppl,         4),
                "lr":         round(current_lr,  8),
                "improved":   improved,
            })
            self._save_history()

            print(f"  └{'─' * 48}\n")

        self._print_summary()

    def _print_summary(self):
        print(f"\n{'═' * 55}")
        print(f"  🏁 Concluído!")
        print(f"  {'Epoch':>6} │ {'Train':>8} │ {'Val':>8} │ {'PPL':>7}")
        print(f"  {'─' * 6}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 7}")

        for h in self.history["epochs"]:
            mark = " ⭐" if h["improved"] else ""
            print(
                f"  {h['epoch']:>6} │ "
                f"{h['train_loss']:>8.4f} │ "
                f"{h['val_loss']:>8.4f} │ "
                f"{h['ppl']:>7.2f}"
                f"{mark}"
            )

        print(f"\n  💾 Melhor Val Loss : {self.best_val_loss:.4f}")
        print(f"  📊 Melhor PPL      : {math.exp(self.best_val_loss):.2f}")
        print(f"{'═' * 55}\n")