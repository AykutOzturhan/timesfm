"""
TimesFM 2.5 LoRA Fine-Tuning — S&P 500 üzerinde.
HuggingFace Transformers + PEFT kullanır.
Model: google/timesfm-2.5-200m-transformers
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from config import (DATA_DIR, CONTEXT_LEN, HORIZON_LEN,
                    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LR, EPOCHS, BATCH_SIZE)

logger = logging.getLogger(__name__)
ADAPTER_DIR = DATA_DIR / "lora_adapter"


class PriceWindowDataset(Dataset):
    """TimesFM için fiyat penceresi dataseti."""
    def __init__(self, X_price: np.ndarray, y: np.ndarray,
                 horizon: int = HORIZON_LEN):
        self.X = torch.tensor(X_price, dtype=torch.float32)
        self.y = torch.tensor(y,       dtype=torch.float32)
        self.horizon = horizon

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        return {"past_values": self.X[i], "future_values": self.y[i]}


def load_transformers_model(model_id: str = "google/timesfm-2.5-200m-transformers"):
    """HuggingFace Transformers versiyonunu yükler."""
    from transformers import AutoModelForSeq2SeqLM, AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_id)
        model  = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return model, config
    except Exception as e:
        logger.error(f"Model yuklenemedi: {e}")
        logger.info("Alternatif: timesfm pytorch modeli kullaniliyor")
        return None, None


def apply_lora(model, rank: int = LORA_RANK,
               alpha: int = LORA_ALPHA,
               dropout: float = LORA_DROPOUT):
    """PEFT LoRA adaptörü uygular."""
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    )
    return get_peft_model(model, lora_cfg)


class TimesFMLoRATrainer:
    """
    S&P 500 verisiyle TimesFM LoRA fine-tuning.

    Not: TimesFM'in transformers versiyonu gereklidir.
    Eğer o versiyon indirilemezse, PyTorch versiyonunu
    direkt fine-tune etmek için torch_finetune() kullanılır.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model  = None

    def setup(self, model_id: str = "google/timesfm-2.5-200m-transformers") -> bool:
        """Model ve LoRA kurulumu. Başarılıysa True döner."""
        try:
            model, _ = load_transformers_model(model_id)
            if model is None:
                return False
            self.model = apply_lora(model)
            self.model.to(self.device)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.model.parameters())
            print(f"  LoRA param: {trainable:,} / {total:,} "
                  f"({trainable/total*100:.2f}% egitilecek)")
            return True
        except Exception as e:
            print(f"  LoRA kurulumu basarisiz: {e}")
            return False

    def train(self, X_price: np.ndarray, y: np.ndarray,
              epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> list[float]:
        """Fine-tuning döngüsü."""
        if self.model is None:
            raise RuntimeError("Önce setup() çağırın")

        dataset = PriceWindowDataset(X_price, y)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim   = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        losses = []

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False):
                past = batch["past_values"].to(self.device)
                fut  = batch["future_values"].to(self.device)
                optim.zero_grad()
                try:
                    out  = self.model(input_values=past, labels=fut.unsqueeze(-1))
                    loss = out.loss
                except Exception:
                    # Basit MSE fallback
                    pred = self.model(past).logits.squeeze(-1)
                    loss = torch.nn.functional.mse_loss(pred[:, -1], fut)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg = epoch_loss / len(loader)
            losses.append(avg)
            print(f"  Epoch {epoch+1}: loss={avg:.5f}")

        return losses

    def save(self, path=ADAPTER_DIR):
        """LoRA adaptörünü kaydeder."""
        if self.model is not None:
            self.model.save_pretrained(str(path))
            print(f"  Adapter kaydedildi: {path}")

    def load(self, base_model_id: str, path=ADAPTER_DIR):
        """Kaydedilmiş adaptörü yükler."""
        from peft import PeftModel
        model, _ = load_transformers_model(base_model_id)
        if model:
            self.model = PeftModel.from_pretrained(model, str(path))
            self.model.to(self.device)
