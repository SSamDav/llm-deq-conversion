import lightning as L
import os
import torch

from datasets import concatenate_datasets, load_dataset
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.optimization import get_wsd_schedule
from jsonargparse import CLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import grad_norm
from llm_deq_conversion.model import DEQLlamaForCausalLM, DEQCausalLMOutputWithPast

class CausalLLM(L.LightningModule):
    def __init__(
        self,
        max_steps,
        model,
        lr=5e-5,
        num_warmup_steps=100,
        num_decay_steps=200,
        weight_decay=0.01,
    ):
        super().__init__()
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.num_decay_steps = num_decay_steps
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.save_hyperparameters()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def training_step(self, batch, batch_idx):
        output= self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        self.log("train_loss", output.loss, on_step=True)
        if isinstance(output, DEQCausalLMOutputWithPast) and output.distance is not None:
            self.log("train_distance", output.distance, on_step=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        ).loss
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_decay_steps=self.num_decay_steps,
            num_training_steps=self.max_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log('Grad Norm', norms[f'grad_2.0_norm_total'],on_step=True, on_epoch=False)


def train(
    datasets: list[dict],
    model_name: str,
    batch_size: int,
    lr: float,
    max_length: int = 1024,
    num_warmup_steps: int = 100,
    num_decay_steps: int = 200,
    weight_decay: float = 0.01,
    deq_max_steps: int = 4,
    phantom_steps: int = 1,
    distance_loss_weight: float = 0.001,
    trainer_args: Optional[dict] = None,
    ckpt_path: Optional[str] = None,
    state_dict_path: Optional[str] = None,
):
    trainer_args = trainer_args or {}
    max_steps = trainer_args["max_steps"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert sum([d["weight"] for d in datasets]) == 1.0, "Dataset weights should sum to 1.0!"
    train_dataset = []
    for d in datasets:
        # "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"
        num_datapoints = d["weight"] * max_steps
        dataset  = load_dataset(**d["args"]) # .select(range(num_datapoints))
        train_dataset.append(dataset)
    train_dataset = concatenate_datasets(train_dataset).shuffle(seed=42, buffer_size=1000)
    
    folder_name = f"deq_steps={deq_max_steps}-phantom_steps={phantom_steps}"     
        
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
        collate_fn=data_collator,
        num_workers=os.cpu_count() 
    )


    # --- Load the model ---
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False

    if deq_max_steps > 0:
        print("Training a DEQ model!!!")
        model = DEQLlamaForCausalLM(config, max_steps=deq_max_steps, phantom_steps=phantom_steps, distance_loss_weight=distance_loss_weight)
        if state_dict_path is None:
            original_model_params  = AutoModelForCausalLM.from_pretrained(model_name).state_dict()
            _ = model.load_state_dict(original_model_params, strict=False)
            del original_model_params
        else:
            _ = model.load_state_dict(torch.load(state_dict_path), strict=False)
        # model.model.gradient_checkpointing = True
        
    else:
        print("Training a normal model!!!")
        model = AutoModelForCausalLM.from_config(config)

    model.gradient_checkpointing_enable()
    lightning_model = CausalLLM(
        model=model,
        max_steps=max_steps,
        lr=lr,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        weight_decay=weight_decay,
    )
    wandb_logger = WandbLogger(project="LLM-to-DEQ", log_model=False)
    wandb_logger.log_hyperparams({**trainer_args, "batch_size": batch_size, "deq_max_steps": deq_max_steps, "phantom_steps": phantom_steps, "distance_weight": distance_loss_weight, "max_length": max_length, "datasets": datasets})
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=max_steps // 4, 
        dirpath=f"checkpoints/{folder_name}",  
        save_last=True,
        filename="{step:06d}-{train_loss:.2f}",  
    )
    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, learning_rate_callback],
        **trainer_args
    )
    trainer.fit(
        lightning_model, train_dataloaders=train_dataloader, ckpt_path=ckpt_path
    )
    torch.save(
        lightning_model.model.state_dict(), f"checkpoints/{folder_name}/final_model.pth"
    )

if __name__ == "__main__":
    CLI(train)
