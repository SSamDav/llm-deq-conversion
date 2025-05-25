import torch

def fill_until(input_ids, max_length, fill_in_value):
    # Create a mask indicating the positions to be filled with eos_token_id
    max_len = input_ids.size(1)
    row_indices = torch.arange(input_ids.size(0), device=input_ids.device).unsqueeze(1)
    col_indices = torch.arange(max_len, device=input_ids.device).unsqueeze(0)
    mask = (col_indices <  max_length.unsqueeze(1))

    # Fill the masked positions with the eos_token_id
    input_ids = input_ids.masked_fill(mask, fill_in_value)
    return input_ids
