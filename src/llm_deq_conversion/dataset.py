from datasets import load_dataset as huggingface_load_dataset

from llm_deq_conversion.utils import fill_until


def load_dataset(
    name: str,
    tokenizer,
    cot: bool = False,
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
        lambda example: {"prompt_length": len(tokenizer(example["question"] + "\n", add_special_tokens=False)["input_ids"])}
    ).map(
        lambda example: tokenizer(example["text"], add_special_tokens=False)
    ).map(
        lambda example: {"input_ids": example["input_ids"] + [tokenizer.eos_token_id], "attention_mask": example["attention_mask"] + [1]},
        remove_columns=drop_columns
    ).map(lambda example: {"answer_length": len(example["input_ids"]) - example["prompt_length"]})
    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from transformers.data.data_collator import DataCollatorForLanguageModeling
    from llm_deq_conversion.utils import fill_until

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", padding_side='left', padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("gsm8k", tokenizer, False)
    data_collator_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(dataset[0])
    batch = data_collator_fn([dataset[0], dataset[1]])
    input_ids = batch["input_ids"]
    answer_len = batch["answer_length"]
    
    print(batch)
    for i, l in enumerate(batch["answer_length"]):
        batch["input_ids"][i, :-l] = tokenizer.eos_token_id
        print(tokenizer.decode(batch["input_ids"][i]))


    input_ids = fill_until(input_ids, answer_len, tokenizer.eos_token_id)
    print(tokenizer.batch_decode(input_ids))
