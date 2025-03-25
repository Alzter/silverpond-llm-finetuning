import pytest
import finetune as ft
from datasets import load_dataset, ClassLabel

@pytest.fixture()
def full_dataset():
    return load_dataset("fancyzhx/dbpedia_14")

@pytest.fixture
def test_dataset():
    return load_dataset("fancyzhx/dbpedia_14", split="test")

def test_preprocess_datadict(full_dataset):
    subsets = ["train", "test"]
    processed_datadict = ft.preprocess_dataset(full_dataset, "content", "label")

    for subset in subsets:
        assert processed_datadict[subset].column_names == ["messages"]

def test_preprocess_dataset(test_dataset):
    samples = test_dataset['content']
    labels = test_dataset['label']
    
    # Translate label IDs to names
    if type(test_dataset.features['label']) is ClassLabel:
        label_names = test_dataset.features['label'].names
        labels = [label_names[label] for label in labels]

    processed_dataset = ft.preprocess_dataset(test_dataset, "content", "label")

    assert processed_dataset.column_names == ["messages"], "Dataset must be in conversational format"

    assert samples == [message[0]['content'] for message in processed_dataset['messages']], "The content of the pre-processed dataset should be identical"
    assert labels == [message[-1]['content'] for message in processed_dataset['messages']], "The content of the pre-processed dataset should be identical"
    