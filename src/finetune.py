from datasets import Dataset, Value, ClassLabel, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
import numpy as np
import pandas as pd
import transformers, torch
from pandas import DataFrame
from copy import copy
import math
import warnings
import json
import os

transformers.set_seed(42) # Enable deterministic LLM output

def create_dataset_from_dataframe(df : DataFrame, text_columns : str | list, label_columns : str | list, test_size : float | None = 0.1, encode_labels : bool = True, dropna : bool = True) -> Dataset:
    """
    Convert a DataFrame into a Dataset for pre-processing.

    Args:
        df (DataFrame): The DataFrame to convert.
        text_columns (str | list): The column name(s) for the input text column (X).
        label_columns (str | list): The column name(s) for the output label column (y). Labels *must* be strings, not class IDs.
        test_size (float, optional): If specified, splits the dataset into train and test subsets where test_size is the ratio of the test subset. Defaults to 0.1.
        encode_labels (bool, optional): If true, converts all class label columns in the Dataset to ClassLabel data type. Defaults to True.
        dropna (bool, optional): If true, removes all rows which contain any null values. Defaults to True.

    Returns:
        Dataset: The dataset.
    """
    df = df.copy() # Copy the dataframe to prevent the original being modified

    if type(label_columns) is str: label_columns = [label_columns]
    if type(text_columns) is str: text_columns = [text_columns]

    # Remove all leading/trailing whitespace within the class labels.
    for label in label_columns:
        df[label] = df[label].map(lambda x : x.strip() if type(x) is str else x)

    # Delete any empty values.
    if dropna: df = df.dropna(subset=[*text_columns, *label_columns])
    
    data = {}

    for feature in text_columns:
        data[feature] = df[feature].to_list()

    for label in label_columns:
        data[label] = df[label].to_list()

    ds = Dataset.from_dict(data)

    if encode_labels:
        for label in label_columns:
            ds = ds.class_encode_column(label) # Convert labels from Value to ClassLabel

    if test_size: # If test_size is not None and is > 0
        ds = ds.train_test_split(test_size=test_size)

    return ds

def select_top_n_classes(dataset : Dataset | DatasetDict, n : int = 10, label_columns : str | list = "label", main_subset : str = 'train', top_n_labels : dict | None = None) -> Dataset:
    """
    Given a dataset, limit the total number of classes to the top ``n`` most common.
    All samples not within the top ``n`` classes are removed.
    
    NOTE: This does *not* adjust the number of samples *per* class to be equal; use ``undersample_dataset()`` for that.

    Args:
        dataset (Dataset | DatasetDict): The dataset to sample.
        n (int, optional): Top ``n`` most common classes to select from the dataset. Defaults to 10.
        label_columns (str | list, optional): The column name(s) for the labels in the dataset. Defaults to "label".
        main_subset (str, optional): If a ``DatasetDict`` is passed to this function (e.g., a dataset split into train/test subsets),
                                        this argument specifies *which* subset to use to get the top ``n`` classes.
                                        Defaults to 'train'.
        top_n_labels (dict | None, optional): Optional arbitrary list of class labels to use. If provided, selects these class labels instead of the top ``n`` classes. Defaults to None.

    Returns:
        Dataset: The dataset with only samples from the top ``n`` classes.
    """

    if type(label_columns) is str: label_columns = [label_columns]

    if top_n_labels is None:
        # Select the top n class labels from the main dataset
        if type(dataset) is DatasetDict:
            # If the dataset is split into train/test sets,
            # use the train subset by default, since that is the largest
            main_dataset = dataset[main_subset]
        else: main_dataset = dataset
        
        n = max(n, 1) # Ensure n >= 1
        # Ensure n is <= maximum number of class labels
        for label in label_columns:
            labels = pd.Series(main_dataset[label])

            num_labels = labels.unique().size
            if n > num_labels:
                warnings.warn(f"\nCannot select top {n} classes from the dataset because the label {label} only has {num_labels} classes.\nProceeding with {num_labels} classes...\n")
                n = num_labels
        
        # Get the top n labels from the dataset as a list
        
        top_n_labels = {}
        for label in label_columns:
            labels = pd.Series(main_dataset[label])
            top_n_labels[label] = labels.value_counts().iloc[0:n].keys().to_list()
    
    # If the dataset is actually a container of datasets,
    # use recursion to sample all sub-datasets
    if type(dataset) is DatasetDict:
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset] = select_top_n_classes(dataset[subset], label_columns=label_columns, n=n, top_n_labels = top_n_labels)
        #dataset = dataset.flatten_indices() # Call .flatten_indices() after .filter() otherwise .sort() takes ages.
        return dataset
    
    for label in label_columns:
        dataset = dataset.filter(lambda x : x[label] in top_n_labels[label])

        # If the labels column is a ClassLabel, we also have to update the label names to match the new classes.
        if type(dataset.features[label]) is ClassLabel:
            
            # Convert class label column into raw strings
            dataset, _ = class_decode_column(dataset, label)
            
            # Cast class label column back into a ClassLabel.
            dataset = dataset.class_encode_column(label)

    #dataset = dataset.flatten_indices() # Call .flatten_indices() after .filter() otherwise .sort() takes ages.
    return dataset

def _get_n_samples_per_class(dataset : Dataset, n : int, label_columns : str | list, shuffle:bool=True, seed:int=0) -> Dataset:
    """
    Given a dataset, obtain a smaller dataset containing the first **n** samples from each class.

    Args:
        dataset (Dataset): The dataset to sample.
        n (int): How many samples from each class to extract.
        label_columns (str | list): The column name(s) for the labels in the dataset.
        shuffle (bool): Whether to sort the final result by class or randomly.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        Dataset: The sample of the dataset.
    """

    if type(label_columns) is str: label_columns = [label_columns]

    for column in [*label_columns]:
        if column not in dataset.features.keys(): raise ValueError(f"Dataset has no column: {column}")
    
    ds_sorted = dataset
    ds_sorted = ds_sorted.flatten_indices().sort(label_columns)
    
    n = max(n, 1) # You must select at least one sample

    class_indices = []
    ds_subset = ds_sorted

    # For each label:
    for label in label_columns:
        
        # Get the number of samples from the least common class.
        num_samples_in_minority_class = pd.Series(ds_subset[label]).value_counts().min()

        # Undersample if needed to ensure an equal number of samples per class.
        if n > num_samples_in_minority_class:
            warnings.warn(f"\nCannot sample {n} samples per class for label {label} equally because some classes have fewer samples.\nSampling {num_samples_in_minority_class} samples per class instead.\n")
            n = min(n, num_samples_in_minority_class)
            n = max(n, 1)

        # Get the index of the first occurence of every unique class in the label
        _, first_class_indices = np.unique(ds_subset[label], return_index=True)

        # For each class, obtain n evenly spaced samples
        label_class_indices = []

        for i, index_start in enumerate(first_class_indices):
            
            # The last sample of each class is the first sample of the next class - 1.
            if i < len(first_class_indices) - 1:
                index_end = first_class_indices[i + 1] - 1
            # The last sample of the last class is the last item in the dataset.
            else:
                index_end = len(ds_subset[label]) - 1

            if index_end < index_start: continue
            label_class_indices.extend(
                
                # Obtain n samples evenly distributed
                # from the first sample of the class to
                # the last sample of the class.
                np.linspace( index_start, index_end,
                    n, endpoint=True, dtype='int'
                ).tolist()
            )

        # Restrict the search space of future labels to
        # only the samples we just selected.
        ds_subset = ds_subset.select(label_class_indices)

        # Append the class indices we just selected to a list.
        class_indices.append(label_class_indices)
    
    # The list of class indices we just produced is hierarchical
    # Each element points to the indices of each previous element
    # To resolve this, we must flatten the nested list like so:

    # Iterate over the list of class indices in reverse order
    for i in reversed(range(len(class_indices))):
        index = class_indices[i]

        # For each index, iterate through every previous index
        for j in reversed(range(i)):

            # Update the index to use the elements of the previous index
            next_index = class_indices[j]
            index = [next_index[k] for k in index]

        class_indices[i] = index

    # Flatten the list of class indices
    class_indices = [i for j in class_indices for i in j]
    # Remove all duplicate indices
    class_indices = np.unique(class_indices).tolist()

    sample = ds_sorted.select(class_indices)

    if shuffle: sample = sample.shuffle(seed=seed)
    
    return sample

def undersample_dataset(dataset : Dataset | DatasetDict, label_columns : str | list = "label", ratio : float = None, size : int = None, samples_per_class : int = None, shuffle : bool = True, seed:int=0) -> Dataset:
    """
    Given a dataset, return a smaller dataset with an equal number of samples per class.
    
    There are 3 methods of sampling a dataset:
    - **ratio**: Specify the size of the sample as a percentage of the original dataset from 1 - 0.
    - **size**: Specify how many items the sample should have in total.
    - **samples_per_class**: Specify how many items each class should have inside the sample.

    Args:
        dataset (Dataset | DatasetDict): The dataset to sample.
        label_columns (str : list, optional): The column name(s) for the labels in the dataset. Defaults to "label".
        ratio (float, optional): What percentage of the dataset to sample from 1-0. Defaults to None.
        size (int, optional): Number of items the new dataset should have. Defaults to None.
        samples_per_class (int, optional): Number of items per class the new dataset should have. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset before and after sampling it. Defaults to True.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        Dataset: The sample of the dataset.
    """

    # If the dataset is actually a container of datasets,
    # use recursion to preprocess all sub-datasets
    if type(dataset) is DatasetDict:
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset] = undersample_dataset(dataset[subset], label_columns, ratio, size, samples_per_class, shuffle, seed)
        return dataset

    if shuffle: dataset=dataset.shuffle(seed=seed)
    
    if type(label_columns) is str: label_columns = [label_columns]

    for column in [*label_columns]:
        if column not in dataset.features.keys(): raise ValueError(f"Dataset has no column: {column}")

    if ratio is None and size is None and samples_per_class is None:
        raise ValueError("Either ratio, size, or samples_per_class must be given.")

    # If samples_per_class is not given, we have to calculate
    # how many samples to allocate to each class based on
    # the size or ratio
    if samples_per_class is None:
        if size is not None:
            ratio = size / dataset.num_rows
        ratio = max(ratio, 0)
        ratio = min(ratio, 1)

        num_labels = 0
        for label in label_columns:
            num_labels += np.unique(dataset[label]).size
        
        samples_per_class = dataset.num_rows / num_labels * ratio
        samples_per_class = int(math.floor(samples_per_class))

    return _get_n_samples_per_class(dataset, samples_per_class, label_columns, shuffle=shuffle, seed=seed)

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

def _sample_label_to_string(sample : dict, class_label_names : list, label_column : str = "completion") -> dict:
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
    sample[label_column] = class_label_names[ int(sample[label_column]) ]
    return sample

def class_decode_column(dataset : Dataset, label_columns : str | list, strip : bool = True) -> tuple[Dataset, dict[str, list[str]]]:
    """Given a Dataset, cast a given column from a ClassLabel to a Value with string data type.

    Args:
        dataset (Dataset): The dataset.
        label_columns (str): Which column(s) to convert from ClassLabel to string.
        strip (bool, optional): Whether to strip the list of all label names to remove duplicates. Defaults to True.

    Returns:
        dataset (Dataset): The dataset with label_columns converted from ClassLabel to string.
        label_names (dict[str, list[str]]): List of all unique class names for each label.
                                            E.g., ``label_names["fruit"] = ["Apple", "Banana", "Orange"]``.
    """
    
    if type(label_columns) is str: label_columns = [label_columns]
    
    label_names = {}
    for label in label_columns:
        # Map the class label column from integer to string.
        if type(dataset.features[label]) is ClassLabel:
            class_names = dataset.features[label].names
            if strip: class_names = [n.strip() for n in class_names]

            # Cast label column from int to str.
            dataset = dataset.cast_column(label, Value(dtype='string'))

            # Replace all class label IDs with label names.
            dataset = dataset.map( lambda sample : _sample_label_to_string(sample, class_names, label_column=label) )
        else:
            class_names = dataset[label]
            if strip: class_names = [n.strip() for n in class_names]
            class_names = list(np.unique(class_names))
        
        label_names[label] = list(class_names)
    
    return dataset, label_names

def _combine_columns_as_json(sample : dict, columns_to_combine : list[str], new_column_name : str = "text") -> dict:
    """Given a row from a Dataset, combine many columns into a new column representing the columns as a JSON string.

    Args:
        sample (dict): The data sample.
        columns_to_combine (list[str]): List of column names to combine.
        new_column_name (str, optional): The name of the new combined column. Defaults to "text".

    Returns:
        dict: The row with combined columns.
    """
    combined = {col: sample[col] for col in columns_to_combine}
    return {new_column_name: json.dumps(combined)}

def combine_columns(dataset : Dataset | DatasetDict, columns : list, new_column_name = "text", delete_columns : bool = True) -> Dataset | DatasetDict:
    """Given a Dataset, combine many columns into one new column where each entry is a JSON string with the values of those columns.

    Args:
        dataset (Dataset | DatasetDict): The dataset.
        columns (list): The list of columns to combine.
        new_column_name (str, optional): The name of the new combined column. Defaults to "text".
        delete_columns (bool, optional): Whether to delete the original columns after combination. Defaults to True.

    Returns:
        Dataset | DatasetDict: The dataset with all columns combined into one.
    """

    # If the dataset is actually a container of datasets,
    # use recursion to preprocess all sub-datasets
    if type(dataset) is DatasetDict:
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset] = combine_columns(dataset[subset], columns, new_column_name, delete_columns)
        return dataset

    dataset = dataset.map(lambda x: _combine_columns_as_json(x, columns, new_column_name))
    if delete_columns: dataset = dataset.remove_columns(columns)

    return dataset

def preprocess_dataset(dataset : Dataset | DatasetDict, text_columns : str | list, label_columns : str | list = "label") -> tuple[Dataset, dict]:
    """
    Pre-process a supervised text-classification dataset into a format usable for fine-tuning.

    Args:
        dataset (Dataset | DatasetDict): A supervised text-classification dataset.
        text_columns (str | list): The column name(s) for the input text column (X).
        label_columns (str | list, optional): The column name(s) for the output label column (y). Defaults to "label".
    
    Returns:
        formatted_dataset (Dataset): The dataset in conversational format.
        label_names (dict): List of all unique class names for each label.
                            E.g., ``label_names["fruit"] = ["Apple", "Banana", "Orange"]``.
    """
    
    # If the dataset is actually a container of datasets,
    # use recursion to preprocess all sub-datasets
    if type(dataset) is DatasetDict:
        label_names = []
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset], new_labels = preprocess_dataset(dataset[subset], text_columns, label_columns)
            if len(new_labels) > len(label_names): label_names = new_labels
        return dataset, label_names

    # Convert text_columns into a list if it is not already
    if type(text_columns) is str: text_columns = [text_columns]
    if type(label_columns) is str: label_columns = [label_columns]

    for column in [*text_columns, *label_columns]:
        if column not in dataset.features.keys(): raise ValueError(f"Dataset has no column: {column}")

    # Select only the text and label columns.
    dataset = dataset.select_columns([*text_columns,*label_columns])
    
    # If there are multiple input text features, combine them into one using JSON formatting
    if len(text_columns) > 1:
        dataset = combine_columns(dataset, text_columns, new_column_name="text")
        
        # Convert text_columns into a string pointing to the combined column
        text_columns = "text"
    else:
        # Convert text_columns into a string pointing to the combined column
        text_columns = text_columns[0]

    # Map the class label column(s) from integer to string.
    dataset, label_names = class_decode_column(dataset, label_columns)

    # If there are multiple labels, combine them into one using JSON formatting
    if len(label_columns) > 1:
        dataset = combine_columns(dataset, label_columns, new_column_name="label")
        label_columns = "label"
    else:
        label_columns = label_columns[0]

    # Convert the dataset into prompt/completion format.
    dataset = dataset.rename_column(text_columns, "prompt")
    dataset = dataset.rename_column(label_columns, "completion")

    # Convert the dataset into conversational format
    dataset = dataset.map(_format_dataset).remove_columns(['prompt', 'completion'])

    return dataset, label_names

def finetune(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, train_dataset : Dataset, lora_config : LoraConfig, sft_config : SFTConfig, output_dir : str | None = None) -> DataFrame:
    """Fine-tune an LLM using LoRA and save the resulting adapters in ``output_dir``. The LLM specified in ``model`` **will** be modified by this function.

    Args:
        model (AutoModelForCausalLM): The LLM to fine-tune, which will be modified by this function. Use ``AutoModelForCausalLM.from_pretrained(model_name)`` to instantiate.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.
        train_dataset (Dataset): The dataset of training samples to fine-tune the model on. You must pre-process this dataset using ``preprocess_dataset`` before calling this method.
        lora_config (LoraConfig): LoRA hyperparameters, including the rank of the adapters and the scaling factor.
        sft_config (SFTConfig): Fine-tuning training configuration, including number of epochs, checkpoints, etc.
        output_dir (str, optional): Where to save the fine-tuned model to. Defaults to ``sft_config.output_dir``.
    
    Returns:
        result (DataFrame): The training history as a DataFrame. The columns are ["Step", "Training Loss"], where "Step" is the epoch and "Training Loss" is the loss value.
    """

    if type(model) is AutoPeftModelForCausalLM:
        raise Exception("Cannot finetune model because it is already finetuned. Merge the adapters into base model to train further.")
    
    if output_dir is None:
        output_dir = sft_config.output_dir

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

    try:
        result = pd.DataFrame(trainer.state.log_history)
        return result
    except Exception as e:
        warnings.warn(f"Error saving training results for model.\n{str(e)}")
        return None

def load_llm(model_name_or_path : str, device_map : str = "cuda:0", quantized : bool = True) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load an LLM from disk or from huggingface.co/models.

    Args:
        model_name_or_path (str): Name of the model.
        device_map (str, optional): Which device to load the fine-tuned model onto. Defaults to "cuda:0".
        quantized (bool, optional): Whether to load the model with 4-bit quantization. Defaults to True.

    Returns:
        model (AutoPeftModelForCausalLM): The LLM.
        tokenizer (AutoTokenizer): The tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    ) if quantized else None

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM(model_name_or_path, device_map=device_map, quantization_config=bnb_config)

    return (model, tokenizer)

def load_finetuned_llm(model_directory : str, device_map : str = "cuda:0", quantized:bool = True) -> tuple[AutoPeftModelForCausalLM, AutoTokenizer]:
    """
    Load a finetuned LLM from disk.

    Args:
        model_directory (str): Where to load the fine-tuned model.
        device_map (str, optional): Which device to load the fine-tuned model onto. Defaults to "cuda:0".
        quantized (bool, optional): Whether to load the model with 4-bit quantization. Defaults to True.

    Returns:
        model (AutoPeftModelForCausalLM): The fine-tuned LLM.
        tokenizer (AutoTokenizer): The tokenizer (unchanged from the base model).
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    ) if quantized else None

    config = PeftConfig.from_pretrained(model_directory)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoPeftModelForCausalLM.from_pretrained(model_directory, device_map=device_map, quantization_config=bnb_config)

    return (model, tokenizer)

def _format_prompt(prompt : str | dict, tokenizer : AutoTokenizer) -> str:
    """
    Convert an LLM prompt into string format with a chat template
    and special tokens.

    Args:
        prompt (str | dict): The prompt for the LLM.
                            You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                            if you want to include a system prompt and/or prior chat history.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.

    Returns:
        prompt (str): The prompt with chat template applied converted to string format using special tokens.
    """
    if type(prompt) is str:
        prompt = [{"role": "user", "content": prompt}]
    
    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    return prompt

def generate(
    prompt : str | dict,
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    max_new_tokens : int = 64,
    response_only : bool = True,
    skip_special_tokens : bool = True,
    do_sample : bool = False,
    temperature : float | None = None,
    top_p : float | None = None,
    top_k : float | None = None,
    kwargs : dict = {}
    ) -> str:
    """
    Generate an LLM response to a given query.

    Args:
        prompt (str | dict): The prompt for the LLM.
                            You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                            if you want to include a system prompt and/or prior chat history.
        model (AutoModelForCausalLM): The LLM to use. Use ``AutoModelForCausalLM.from_pretrained(model_name)`` to instantiate.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.
        max_new_tokens (int, optional): Maximum number of tokens for the model to output. Defaults to 64.
        response_only (bool, optional): If True, excludes all previous messages from the output. Defaults to True.
        skip_special_tokens (bool, optional): If True, removes model special tokens from the output. Defaults to True.
        do_sample (bool, optional): If False, enables deterministic generation. Defaults to False.
        temperature (float, optional): Higher = greater likelihood of low probability words. Leave empty if ``do_sample`` is False. Defaults to None.
        top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if ``do_sample`` is False. Defaults to None.
        top_k (float, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Leave empty if ``do_sample`` is False. Defaults to None.
        kwargs (dict, optional): Additional parameters to pass into ``model.generate()``. Defaults to {}.
    Returns:
        response (str): The LLM's response.
    """

    # Convert user query into a formatted prompt
    prompt = _format_prompt(prompt, tokenizer=tokenizer)

    # Tokenize the formatted prompt
    tokenized_input = tokenizer(prompt,
                                add_special_tokens=False,
                                return_tensors="pt").to(model.device)
    model.eval()

    # Generate the response
    generation_output = model.generate(**tokenized_input,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=do_sample,
                                       temperature=temperature,
                                       top_p = top_p,
                                       top_k = top_k,
                                       **kwargs)

    # If required, remove the tokens belonging to the prompt
    if response_only:
        input_length = tokenized_input['input_ids'].shape[1]
        generation_output = generation_output[:, input_length:]
    
    # Decode the tokens back into text
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=skip_special_tokens)[0]
    return output
