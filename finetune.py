from datasets import Dataset, Value, ClassLabel, DatasetDict
import numpy as np

def _get_n_samples_per_class(dataset : Dataset, n : int, shuffle:bool=True, seed:int=0) -> Dataset:
    """
    Given a dataset, obtain a smaller dataset containing the first **n** samples from each class.

    Args:
        dataset (Dataset): The dataset to sample.
        n (int): How many samples from each class to extract.
        shuffle (bool): Whether to sort the final result by class or randomly.
        seed (int): RNG seed.

    Returns:
        Dataset: The sample of the dataset.
    """
    ds_sorted = dataset.sort('label')
    _, class_indices = np.unique(ds_sorted['label'], return_index=True)

    class_indices = np.array([list(range(index, index + n)) for index in class_indices])
    class_indices = class_indices.flatten()

    sample = dataset.sort('label').select(class_indices)
    if shuffle: sample = sample.shuffle(seed=seed)
    
    return sample

def sample_dataset(dataset : Dataset, ratio : float = None, size : int = None, samples_per_class : int = None) -> Dataset:
    """
    Given a dataset, return a smaller dataset with an equal number of samples per class.
    
    There are 3 methods of sampling a dataset:
    - **ratio**: Specify the size of the sample as a percentage of the original dataset from 1 - 0.
    - **size**: Specify how many items the sample should have in total.
    - **samples_per_class**: Specify how many items each class should have inside the sample.

    Args:
        dataset (Dataset): The dataset to sample.
        ratio (float, optional): What percentage of the dataset to sample from 1-0.
        size (int, optional): Number of items the new dataset should have.
        samples_per_class (int, optional): Number of items per class the new dataset should have.

    Returns:
        Dataset: The sample of the dataset.
    """

    if ratio is None and size is None and samples_per_class is None:
        raise ValueError("Either ratio, size, or samples_per_class must be given.")

    if samples_per_class is None:
        if size is not None:
            ratio = size / dataset.num_rows
        ratio = max(ratio, 0)
        ratio = min(ratio, 1)
    
        samples_per_class = dataset.num_rows // len(dataset.features['label'].names)
        samples_per_class = int(samples_per_class * ratio)

    return _get_n_samples_per_class(dataset, samples_per_class)

def _format_dataset(examples : Dataset) -> dict:
    """
    Converts a Dataset in [prompt/completion format](https://huggingface.co/docs/trl/main/en/dataset_formats#standard)
    to [conversational format](https://huggingface.co/docs/trl/main/en/dataset_formats#conversational).
    This is necessary because the trl library no longer supports prompt/completion datasets for LLM fine-tuning.
    Adapted from [Daniel Voigt Godoy](https://github.com/dvgodoy/FineTuningLLMs/blob/main/Chapter0.ipynb).
    
    Args:
        examples (Dataset): Dataset in prompt/completion format.

    Returns:
        Dataset: Dataset in conversational format.
    """
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

def _label_to_string(sample : dict, class_label_names : list) -> dict:
    """
    Given a sample from a supervised text classification dataset
    in prompt/completion format, replace the class label ID with
    a label name string. The class label column must already have
    a string data type.

    Args:
        sample (dict): The data sample in prompt/completion format with string data type.
        class_label_names (list): A list of all class label names.

    Returns:
        dict: The data sample with class ID replaced with label name.
    """
    sample['completion'] = class_label_names[ int(sample['completion']) ]
    return sample

def preprocess_dataset(dataset : Dataset | DatasetDict, text_column : str, labels_column : str) -> Dataset:
    """
    Pre-process a supervised text-classification dataset into a format usable for fine-tuning.

    Args:
        dataset (Dataset | DatasetDict): A supervised text-classification dataset.
        text_column (str): The column name for the input text column (X).
        labels_column (str): The column name for the output label column (y).
    
    Returns:
        Dataset: The dataset in conversational format.
    """

    # If the dataset is actually a container of datasets,
    # use recursion to preprocess all sub-datasets
    if type(dataset) is DatasetDict:
        for subset in dataset.keys():
            dataset[subset] = preprocess_dataset(dataset[subset], text_column=text_column, labels_column=labels_column)
        return dataset

    # Select only the text and label columns.
    dataset = dataset.select_columns([text_column,labels_column])

    # Convert the dataset into prompt/completion format.
    dataset = dataset.rename_column(text_column, "prompt")
    dataset = dataset.rename_column(labels_column, "completion")

    # Map the class label column from integer to string.
    if type(dataset.features['completion']) is ClassLabel:

        label_names = dataset.features['completion'].names
        # Cast label column from int to str.
        dataset = dataset.cast_column("completion", Value(dtype='string'))
        # Replace all class label IDs with label names.
        dataset = dataset.map( lambda sample : _label_to_string(sample, label_names) )

    # Convert the dataset into conversational format
    dataset = dataset.map(_format_dataset).remove_columns(['prompt', 'completion'])

    return dataset

