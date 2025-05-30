import pytest
import preprocess as pre
from datasets import load_dataset, ClassLabel, Value
import numpy as np
import pandas as pd
import json

@pytest.fixture()
def csv_dataset():
    return pd.read_csv("tests/test_data.csv", low_memory=False)
    
@pytest.fixture()
def full_dataset():
    return load_dataset("fancyzhx/dbpedia_14")

@pytest.fixture
def test_dataset():
    return load_dataset("fancyzhx/dbpedia_14", split="test")

def test_convert_csv_to_dataset(csv_dataset):
    text_column = "Final Narrative"
    labels_column = "NatureTitle"

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_column, labels_column, test_size=0, encode_labels=False, dropna=False)

    assert dataset.column_names == ['Final Narrative', 'NatureTitle'], "The converted dataset should have 'Final Narrative' and 'NatureTitle' features."
    assert len(dataset[text_column]) == len(csv_dataset[text_column]), "The 'text' feature should correspond to the text column of the dataset."
    assert len(dataset[labels_column]) == len(csv_dataset[labels_column]), "The 'label' feature should correspond to the label column of the dataset."

    end = len(dataset) - 1

    assert(dataset[text_column][0] == csv_dataset[text_column][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[labels_column][0] == csv_dataset[labels_column][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[text_column][end] == csv_dataset[text_column][end]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[labels_column][end] == csv_dataset[labels_column][end]), "The Dataset should have the same contents as the DataFrame"

def test_convert_csv_with_multiple_inputs(csv_dataset):
    text_columns = ["Final Narrative", "Inspection"]
    labels_column = "NatureTitle"

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_columns, labels_column, test_size=0, encode_labels=False, dropna=False)

    assert dataset.column_names == ["Final Narrative", "Inspection", "NatureTitle"], "The converted dataset should preserve the multiple input features."

    assert len(dataset[text_columns[0]]) == len(csv_dataset[text_columns[0]]), "Datasets should have same size"
    assert len(dataset[text_columns[-1]]) == len(csv_dataset[text_columns[-1]]), "Datasets should have same size"
    assert len(dataset[labels_column]) == len(csv_dataset[labels_column]), "Datasets should have same size"

    assert(dataset[ text_columns[0] ][0] == csv_dataset[text_columns[0]][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ text_columns[-1] ][0] == csv_dataset[text_columns[-1]][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[labels_column][0] == csv_dataset[labels_column][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ text_columns[0] ][100] == csv_dataset[text_columns[0]][100]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ text_columns[-1] ][100] == csv_dataset[text_columns[-1]][100]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[labels_column][100] == csv_dataset[labels_column][100]), "The Dataset should have the same contents as the DataFrame"

def test_convert_csv_with_multiple_outputs(csv_dataset):

    text_column = "Final Narrative"
    label_columns = ["NatureTitle", "Part of Body Title"]

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_column, label_columns, test_size=0, encode_labels=False, dropna=False)

    assert len(dataset[text_column]) == len(csv_dataset[text_column]), "Datasets should have same size"
    assert len(dataset[label_columns[0]]) == len(csv_dataset[label_columns[0]]), "Datasets should have same size"
    assert len(dataset[label_columns[1]]) == len(csv_dataset[label_columns[1]]), "Datasets should have same size"

    assert(dataset[text_column][0] == csv_dataset[text_column][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ label_columns[0] ][0] == csv_dataset[label_columns[0] ][0]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ label_columns[-1] ][0] == csv_dataset[label_columns[-1] ][0]), "The Dataset should have the same contents as the DataFrame"
    
    assert(dataset[text_column][100] == csv_dataset[text_column][100]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ label_columns[0] ][100] == csv_dataset[label_columns[0] ][100]), "The Dataset should have the same contents as the DataFrame"
    assert(dataset[ label_columns[-1] ][100] == csv_dataset[label_columns[-1] ][100]), "The Dataset should have the same contents as the DataFrame"
    
def test_class_encode_decode(csv_dataset):
    text_column = "Final Narrative"
    labels_column = "NatureTitle"

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_column, labels_column, test_size=0, encode_labels=True, dropna=False)
    dataset, label_names = pre.class_decode_column(dataset, labels_column, strip=False)

    original = list(csv_dataset[labels_column])
    new = list(dataset[labels_column])

    assert len(original) == len(new), "Encoding and then decoding a label column should return it to its original state"
    
    assert original[0] == new[0], "Encoding and then decoding a label column should return it to its original state"
    assert original[-1] == new[-1], "Encoding and then decoding a label column should return it to its original state"
    #assert list(dataset[labels_column]) == list(csv_dataset[labels_column]), "Encoding and then decoding a label column should return it to its original state"

def test_combine_features(test_dataset):
    columns = ["title", "content"]
    combined = pre.combine_columns(test_dataset, columns, new_column_name = "text")

    subset = test_dataset.select_columns(columns)

    assert len(combined["text"]) == len(subset), "The combined column should have the same number of elements as its constituents"

    expected = json.dumps(subset[0])
    actual = combined[0]["text"]

    assert expected == actual, "The contents of the combined column should be a JSON string containing the composite columns"

    expected = json.dumps(subset[-1])
    actual = combined[-1]["text"]

    assert expected == actual, "The contents of the combined column should be a JSON string containing the composite columns"

def test_decode_classlabel(test_dataset):
    assert type(test_dataset.features["label"]) is ClassLabel, "The label column should start as a ClassLabel"
    old_values = test_dataset["label"]

    test_dataset, label_names = pre.class_decode_column(test_dataset, "label")
    label_names = label_names['label']

    assert type(test_dataset.features["label"]) is Value, "Apre.r class decoding, the class label should be a Value with dtype string"
    assert test_dataset.features['label'].dtype == 'string', "Apre.r class decoding, the class label should be a Value with dtype string"
    assert type(test_dataset['label'][0]) is str, "Apre.r class decoding, the class label should be a Value with dtype string"

    new_values = test_dataset["label"]

    assert len(old_values) == len(new_values), "The length of the decoded class labels should be identical to the encoded ones"

    unique_values = list( np.unique( [n.strip() for n in new_values if type(n) is str] ) )

    label_names.sort()
    unique_values.sort()

    print(label_names)
    print(unique_values)

    #tm.assert_series_equal(label_names, csv_dataset[labels_column])
    assert label_names == unique_values, "The list label_names should contain all unique values for the class label"

def test_resize_datadict(full_dataset):
    size = np.array(list(full_dataset.shape.values()))[:,0]

    sample = pre.undersample_dataset(full_dataset, ratio=0.5)

    new_size = np.array(list(sample.shape.values()))[:,0]

    assert np.array_equal(size // 2, new_size)

def test_resize_dataset(test_dataset):
    size = len(test_dataset)

    sample_ratio = pre.undersample_dataset(test_dataset, ratio=0.5)
    assert len(sample_ratio) == size // 2

    sample_size = pre.undersample_dataset(test_dataset, size=1400)
    assert len(sample_size) == 1400

    sample_per_class = pre.undersample_dataset(test_dataset, samples_per_class=10)
    assert len(sample_per_class) == 14 * 10

def test_resize_dataset_requires_method(test_dataset):
    with pytest.raises(ValueError):
        pre.undersample_dataset(test_dataset)

def test_resize_dataset_has_no_side_effects(test_dataset):
    expected = len(test_dataset)

    pre.undersample_dataset(test_dataset, ratio=0.1)

    actual = len(test_dataset)

    assert expected == actual, "There should be no side effects from resizing a dataset"

def test_preprocess_datadict(full_dataset):
    subsets = ["train", "test"]
    full_dataset = pre.undersample_dataset(full_dataset, samples_per_class=10)
    processed_datadict, _ = pre.preprocess_dataset(full_dataset, "content", "label")

    for subset in subsets:
        assert processed_datadict[subset].column_names == ["messages"]

def test_preprocess_dataset(test_dataset):
    samples = test_dataset['content']
    labels = test_dataset['label']
    
    # Translate label IDs to names
    if type(test_dataset.features['label']) is ClassLabel:
        label_names = test_dataset.features['label'].names
        labels = [label_names[label] for label in labels]

    processed_dataset, _ = pre.preprocess_dataset(test_dataset, "content", "label")

    assert processed_dataset.column_names == ["messages"], "Dataset must be in conversational format"

    assert samples == [message[0]['content'] for message in processed_dataset['messages']], "The content of the pre-processed dataset should be identical"
    assert labels == [message[-1]['content'] for message in processed_dataset['messages']], "The content of the pre-processed dataset should be identical"

def test_preprocess_dataset_with_multiple_input_features(test_dataset):

    columns = ["title", "content"]
    processed_dataset, _ = pre.preprocess_dataset(test_dataset, columns, "label")

    assert processed_dataset.column_names == ["messages"], "Dataset must be in conversational format"

    sample = test_dataset.select_columns(columns)

    expected = json.dumps( sample[0] )
    actual = processed_dataset['messages'][0][0]['content']
    assert expected == actual, "The input feature of the preprocessed dataset should be a JSON string representing the several input features that were selected"

    expected = json.dumps( sample[-1] )
    actual = processed_dataset['messages'][-1][0]['content']
    assert expected == actual, "The input feature of the preprocessed dataset should be a JSON string representing the several input features that were selected"

def test_preprocess_dataset_no_side_effects(test_dataset):
    expected = test_dataset.column_names

    pre.preprocess_dataset(test_dataset, "content", "label")

    actual = test_dataset.column_names

    assert expected == actual, "There should be no side effects from resizing a dataset"

def test_get_class_labels(test_dataset):

    expected = {"label" : test_dataset.features['label'].names}
    _, actual = pre.preprocess_dataset(test_dataset, "content", "label")

    assert expected == actual, "A dict of class label names should be returned when preprocessing a dataset."

def test_get_multiple_class_labels(csv_dataset):
    text_column = ["Final Narrative"]
    label_columns = ["NatureTitle", "Part of Body Title"]

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_column, label_columns, test_size=0)

    expected = {
        "NatureTitle" : dataset.features["NatureTitle"].names,
        "Part of Body Title" : dataset.features["Part of Body Title"].names
    }

    _, actual = pre.preprocess_dataset(dataset, text_column, label_columns)

    assert expected == actual, "A dict of class label names should be returned when preprocessing a dataset with multiple labels."

def test_get_top_10_classes(csv_dataset):
    text_column = "Final Narrative"
    labels_column = "NatureTitle"

    # Get the top 10 classes from the CSV dataset
    new_data = csv_dataset.copy()
    new_data = new_data.dropna(subset=labels_column)
    new_data[labels_column] = new_data[labels_column].map(lambda x : x.strip())
    top_classes = new_data[labels_column].value_counts()[0:10].keys().tolist()

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_column, labels_column, test_size=0)
    
    dataset = pre.select_top_n_classes(dataset, n=10, label_columns=labels_column)

    labels = dataset.features[labels_column].names

    # The order doesn't matter
    labels.sort()
    top_classes.sort()

    assert len(labels) == 10, "Getting the top 10 classes should return only 10 classes"
    assert labels == top_classes, "The top 10 classes should be the 10 most common clases"

def test_get_top_10_classes_from_multiple_labels(csv_dataset):
    text_column = "Final Narrative"
    label_columns = ["NatureTitle", "Part of Body Title"]

    top_classes_list = {}
    for label in label_columns:
        # Get the top 10 classes from the CSV dataset
        new_data = csv_dataset.copy()
        new_data = new_data.dropna(subset=label)
        new_data[label] = new_data[label].map(lambda x : x.strip())
        top_classes_list[label] = new_data[label].value_counts()[0:10].keys().tolist()

    dataset = pre.create_dataset_from_dataframe(csv_dataset, text_column, label_columns, test_size=0)
    
    dataset = pre.select_top_n_classes(dataset, n=10, label_columns=label_columns)

    for label in label_columns:
        labels = dataset.features[label].names
        top_classes = top_classes_list[label]

        top_classes.sort()
        labels.sort()

        assert len(labels) == 10, "Getting the top 10 classes should return only 10 classes"
        assert labels == top_classes, "The top 10 classes should be the 10 most common clases"

def test_undersample_multiclass(csv_dataset):
    input_features = ["Final Narrative", "EventTitle"]
    output_labels = ["NatureTitle", "Part of Body Title"]
    # Convert DataFrame into a Dataset
    dataset = pre.create_dataset_from_dataframe(csv_dataset, input_features, output_labels)
    # Select items from the 10 most common classes
    dataset = pre.select_top_n_classes(dataset, n=10, label_columns=output_labels)
    # Sample an equal number of items from each class
    pre.undersample_dataset(dataset, ratio=1,label_columns=output_labels)
