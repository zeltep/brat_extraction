from torch.utils.data import Dataset

class BratTokenClassificationDataset(Dataset):
    def __init__(self, texts, entity_annotations, tokenizer, max_length, label2id):
        self.texts = texts
        self.entity_annotations = entity_annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        entity_indices = self.entity_annotations[idx]

        # Tokenize input
        tokenized_inputs = self.tokenizer(text, return_offsets_mapping=True, padding=True, truncation=True)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        offset_mapping = tokenized_inputs["offset_mapping"]

        # Create token-level labels
        labels = ["O"] * len(input_ids)
        for start, end, label in entity_indices:
            start_token = None
            end_token = None

            for i, (start_offset, end_offset) in enumerate(offset_mapping):
                if start_offset == start:
                    start_token = i
                if end_offset == end:
                    end_token = i

            # Set labels for the entity tokens
            if start_token is not None and end_token is not None:
                labels[start_token:end_token + 1] = ["B-" + label] + ["I-" + label] * (end_token - start_token)

        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        labels = [self.label2id[x] for x in labels]

        labels[0] = -100
        labels += [-100] * padding_length

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}