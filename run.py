from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dtorch.dloader import stream_dataset_from_dir


def collect_texts(dataset_dir: str, max_samples: int = 1000):
    texts = []
    i = 0
    for ex in stream_dataset_from_dir(dataset_dir):
        parts = []
        for v in ex.values():
            if isinstance(v, str):
                parts.append(v)
        if not parts:
            continue
        texts.append("\n".join(parts))
        i += 1
        if i >= max_samples:
            break
    return texts


class TokenBlockDataset(Dataset):
    def __init__(self, ids: List[int], seq_len: int):
        tokens = torch.tensor(ids, dtype=torch.long)
        total = tokens.size(0) // (seq_len + 1)
        self.blocks = tokens[: total * (seq_len + 1)].view(total, seq_len + 1)

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, idx):
        b = self.blocks[idx]
        return b[:-1], b[1:]


def main(base_model: str, dataset_dir: str, seq_len: int, batch_size: int, lr: float, epochs: int, max_samples: int, device: str, save_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    model.to(device)

    texts = collect_texts(dataset_dir, max_samples=max_samples)
    concatenated = "\n\n".join(texts)
    enc = tokenizer(concatenated, return_attention_mask=False)
    ids = enc['input_ids']

    dataset = TokenBlockDataset(ids, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(input_ids=xb, labels=yb)
            loss = outputs.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(dataloader):.4f}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    base_model = 'Qwen/Qwen3-4B-Instruct-2507'
    dataset_dir = 'cache/raw/BelleGroup_school_math_0.25M'
    seq_len = 512
    batch_size = 1
    lr = 2e-5
    epochs = 1
    max_samples = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = 'dtorch/qwen_finetuned'

    main(base_model, dataset_dir, seq_len, batch_size, lr, epochs, max_samples, device, save_path)
