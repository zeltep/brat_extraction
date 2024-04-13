## BRAT2IOB

Extracts BRAT annotations and converts them to IOB format for NLP model training

## Usage

```python
from transformers import AutoTokenizer
from extraction import BRAT2IOB

path = '/path_to_BRAT_dir'

#Create a tokenizer using your chosen model
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

#Define which labels to include in datasePt
accepted_labels = ['Tobacco'] 

#The maximum length of tokens (depends on your model size)
max_length = 350


dataset = BRAT2IOB(path, tokenizer, accepted_labels, max_length)

```


## Contributions

[bratreader](https://github.com/clips/bratreader)
