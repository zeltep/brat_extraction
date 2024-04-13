from bratreader.repomodel import RepoModel
from BratTokenClassificationDataset import BratTokenClassificationDataset

def BRAT2IOB(path, tokenizer, accepted_labels, max_length=350):
    """Extracts formatted BRAT annotations and converts them to IOB format
    :param path: (str) path to BRAT directory containing .ann and .txt files
    :param tokenizer: (transformers.AutoTokenizer) tokenizer for text
    :param accepted_labels: (list) contains labels with which to filter
    :param max_length: (int) max_length with which to tokenizer notes

    returns BratTokenClassificationDataset: inherits torch.utisl.data.Dataset, labeled in IOB format
    """

    notes_repo = RepoModel(path)

    texts = []
    annotations = []
        

    for doc_id, doc in notes_repo.documents.items():
        spans = []

        filtered_anns = [ann for ann in doc.annotations if
                     (list(ann.labels.keys())[0] in accepted_labels)]

        # Remove annotations that have the same spans
        filtered_anns = [ann for i, ann in enumerate(filtered_anns) if 
                        (ann.spans not in [a.spans for a in filtered_anns[(i+1):]])]

        # Combine spans for each entity
        filtered_anns = [(ann.spans[0][0], ann.spans[-1][-1], list(ann.labels.keys())[0]) 
                         for ann in filtered_anns]
        
        texts.append(doc.text)
        annotations.append(filtered_anns)

    
    label_list = ["O"] + [flat_l for l in accepted_labels for flat_l in ["B-"+l, "I-"+l]]

    id2label = dict([(idx, label) for idx, label in enumerate(label_list)])
    label2id = dict([(label, idx) for idx, label in id2label.items()])

    dataset = BratTokenClassificationDataset(texts, annotations, tokenizer, max_length=max_length, label2id=label2id)

    return dataset
    
    

