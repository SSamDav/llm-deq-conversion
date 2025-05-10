import lightning as L
import torch

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
from llm_deq_conversion.model import DEQLlamaForCausalLMV2, DEQCausalLMOutputWithPast
from llm_deq_conversion.dataset import load_dataset

class CausalLLM(L.LightningModule):
    def __init__(
        self,
        max_steps,
        model,
        lr=5e-5,
        num_warmup_steps=100,
        num_decay_steps=200,
        weight_decay=0.01,
        eos_token_id=0,
    ):
        super().__init__()
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.num_decay_steps = num_decay_steps
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.model = model
        self.eos_token_id = eos_token_id
        self.save_hyperparameters(ignore=["model", "eos_token_id"])

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def training_step(self, batch, batch_idx):
        batch["labels"][:, -1] = self.eos_token_id
        output= self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        self.log("train_loss", output.loss, on_step=True)
        if isinstance(output, DEQCausalLMOutputWithPast):
            self.log("train_distance", output.stats["abs_lowest"], on_step=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        batch["labels"][:, -1] = self.eos_token_id
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
    dataset_name: str,
    model_name: str,
    batch_size: int,
    lr: float,
    use_cot: bool = False,
    max_length: int = 1024,
    num_warmup_steps: int = 100,
    num_decay_steps: int = 200,
    weight_decay: float = 0.01,
    deq_max_steps: int = 4,
    phantom_steps: int = 1,
    damp: float = 0.8,
    seed: int = 42,
    trainer_args: Optional[dict] = None,
    ckpt_path: Optional[str] = None,
    continue_training: bool = True,
):
    trainer_args = trainer_args or {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', padding=True, truncation=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(name=dataset_name, tokenizer=tokenizer, cot=use_cot)
    train_dataset = train_dataset.train_test_split(test_size=500)
    val_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    cot_text = "cot" if use_cot else "no-cot" 
    folder_name = f"{dataset_name}[{cot_text}]_deq_steps={deq_max_steps}-phantom_steps={phantom_steps}"     
      
       
    data_collator_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator_fn,
        num_workers=16,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=data_collator_fn,
        num_workers=16,
        shuffle=False
    )
    max_steps = trainer_args.get("max_steps", trainer_args["max_epochs"] * len(train_dataloader))


    # --- Load the model ---
    config = AutoConfig.from_pretrained(model_name)
    if deq_max_steps > 0:
        print("Training a DEQ model!!!")
        # TODO: Fix cache
        config.use_cache = False
        model = DEQLlamaForCausalLMV2(config, max_steps=deq_max_steps, phantom_steps=phantom_steps, damp=damp)
        if ckpt_path is None:
            original_model_params  = AutoModelForCausalLM.from_pretrained(model_name).state_dict()
            _ = model.load_state_dict(original_model_params, strict=False)
            del original_model_params
        elif continue_training is False:
            print(f"Loading weights from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, weights_only=False)
            _ = model.load_state_dict(
                {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
            )
            del ckpt
            ckpt_path = None
    else:
        print("Training a normal model!!!")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.gradient_checkpointing_enable()
    lightning_model = CausalLLM(
        model=model,
        eos_token_id=tokenizer.eos_token_id,
        max_steps=max_steps,
        lr=lr,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        weight_decay=weight_decay,
    )
    wandb_logger = WandbLogger(project="LLM-to-DEQ", log_model=False)
    wandb_logger.log_hyperparams({**trainer_args, "batch_size": batch_size, "deq_max_steps": deq_max_steps, "phantom_steps": phantom_steps, "max_length": max_length, "dataset_name": dataset_name, "use_cot": use_cot, "damp": damp})
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=max_steps // 5,
        save_top_k=-1,
        dirpath=f"checkpoints/{folder_name}",  
        save_last=True,
        filename="{step:06d}-{train_loss:.2f}",  
    )
    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, learning_rate_callback]
    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        **trainer_args
    )
    trainer.fit(
        lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path
    )
    torch.save(
        lightning_model.model.state_dict(), f"checkpoints/{folder_name}/final_model.pth"
    )

if __name__ == "__main__":
    CLI(train)
