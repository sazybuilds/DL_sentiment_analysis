import torch.nn as nn
import torch

def tokenize_and_numericalize_example(example, tokenizer, max_length=None):
    ids = tokenizer(
        example["text"], truncation=True, max_length=max_length
    )["input_ids"]
    return {"ids": ids}


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        attention_mask = batch_ids.ne(pad_index).long()
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {
            "ids": batch_ids,
            "attention_mask": attention_mask,
            "label": batch_label,
        }
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False, pin_memory=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return data_loader
