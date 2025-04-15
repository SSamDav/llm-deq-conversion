import lightning as L
import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchdeq.solver.broyden import broyden_solver
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from jsonargparse import CLI
from lightning.pytorch.loggers import WandbLogger


def select_first_k(dataset, k):
    selected_data = []
    for i, item in enumerate(dataset):
        if i >= k:  # Stop after collecting one shard's worth
            break
        selected_data.append(item)
    return Dataset.from_list(selected_data)


class CausalLLM(L.LightningModule):
    def __init__(self, model, lr=5e-5, warmup_steps=100, weight_decay=0.01, estimated_stepping_batches: Optional[int] = None):
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.estimated_stepping_batches = estimated_stepping_batches
        self.save_hyperparameters()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def training_step(self, batch, batch_idx):
        logits = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # (batch * seq_len, vocab)
            shift_labels.view(-1)                          # (batch * seq_len)
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        val_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # (batch * seq_len, vocab)
            shift_labels.view(-1)                          # (batch * seq_len)
        )
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]    


class DEQCausalLLM(CausalLLM):
    def __init__(self, model, damp: float = 0.9, phantom_steps: int = 3, lr=5e-5, warmup_steps=100, weight_decay=0.01, estimated_stepping_batches: Optional[int] = None):
        self.damp = damp
        self.phantom_steps = phantom_steps
        super().__init__(model, lr, warmup_steps, weight_decay, estimated_stepping_batches)
        
    def forward(self, input_ids, attention_mask, labels=None):
        u = self.model.model.embed_tokens(input_ids)
        z0 = torch.zeros_like(u)
        f = lambda x: self.model.model(inputs_embeds=(u + x), attention_mask=attention_mask).last_hidden_state
        with torch.no_grad():
          h, _, _ = broyden_solver(f, z0)

        start_h = h.clone()
        if self.training:
            for _ in range(self.phantom_steps):
              h = (1 - self.damp) * h + self.damp * f(h)
        distance = (start_h - h).norm(p=1)
        logits = self.model.lm_head(h)
        return logits, distance

    def training_step(self, batch, batch_idx):
        logits, distance  = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # (batch * seq_len, vocab)
            shift_labels.view(-1)                          # (batch * seq_len)
        )
        loss += distance
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, distance = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        val_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # (batch * seq_len, vocab)
            shift_labels.view(-1)                          # (batch * seq_len)
        )
        val_loss += distance
        self.log("val_loss", val_loss)


def train(
    model_name: str,
    batch_size: int,
    lr: float,
    max_length: int = 1024,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    trainer_args: Optional[dict] = None,
    ckpt_path: Optional[str] = None
):
    trainer_args = trainer_args or {}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset= load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    drop_columns = train_dataset.column_names
    tokenize_function = lambda example: tokenizer(example["text"], truncation=True, max_length=max_length)
    train_dataset = (
        train_dataset.map(tokenize_function, remove_columns=drop_columns)
        .with_format("torch")
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing = True
    lightning_model = DEQCausalLLM(
        model=model,
        lr=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        estimated_stepping_batches=trainer_args.get("max_steps")
    )
    wandb_logger = WandbLogger(project="LLM-to-DEQ", log_model=False)
    trainer = L.Trainer(
        logger=wandb_logger,
        **trainer_args
    )
    trainer.fit(
        lightning_model, train_dataloaders=train_dataloader, ckpt_path=ckpt_path
    )

if __name__ == "__main__":
    CLI(train)
