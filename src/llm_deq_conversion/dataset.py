from datasets import load_dataset as huggingface_load_dataset


def load_dataset(
    name: str,
    tokenizer,
    cot: bool = False
):
    dataset = huggingface_load_dataset(
        path="zen-E/GSM8k-Aug",
        split="train"
    ).map(
        lambda example: {"text": example["question"] + "\n"}
    )
    if cot:
        dataset = dataset.map(
            lambda example: {"text": example["text"] + example["cot"] + "\n"}
        )
    dataset = dataset.map(
        lambda example: {"text": example["text"] + "### " + example["answer"]}
    )
    drop_columns = dataset.column_names
    dataset = dataset.map(
        lambda example: tokenizer(example["text"], add_special_tokens=False)
    ).map(
        lambda example: {"input_ids": example["input_ids"] + [tokenizer.eos_token_id], "attention_mask": example["attention_mask"] + [1]},
        remove_columns=drop_columns
    )
    return dataset
