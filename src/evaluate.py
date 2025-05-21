import dataclasses
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal
from datasets import Dataset
from utils import PretrainedLM, ModelResponse
import os, re
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import time, json
from datetime import timedelta
import warnings

def _sanitize_string(string) -> str:
    """Make a given string safe for use within a file path.
    Args:
        string (str): A string.
    Returns:
        str: The file-safe string.
    """
    # Remove all whitespace and make the string lowercase
    string = string.lower().strip().replace(" ", "_")
    # Remove all non-alphanumeric characters
    string = "".join(c for c in string if c.isalnum() or c in ["-", "_", " "])
    return string

@dataclass
class EvaluationConfig:
    """
    Defines what instructions and hyperparameters to give an LLM to classify each sample in a text classification dataset.
    You can specify different configurations for different prompting and generation techniques.

    Args:
        technique_name (str): The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".
        max_tokens (int): How many tokens the LLM is allowed to produce to classify each sample.
                          If you are planning on having your LLM output *just* the class label,
                          you can set this value to 1. The LLM will only return the first few
                          letters of the class label, but this is usually enough to identify
                          which label it selected. See ``_get_class_id_from_model_response()`` for implementation details.
        prompt (str, optional): Optional prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.
        prompt_role (str, optional): What role to give the LLM prompt. Defaults to "system", meaning a system prompt. Can be replaced with "user" for models which do not work well with system prompts.
        temperature (float, optional): Sampling temperature to be used. Higher = greater likelihood of low probability words. Defaults to 1.
        top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if temperature > 0. Defaults to None.
        out_path (str,optional): Which directory to save the evaluation result by default. Defaults to "results".
        """
    technique_name : str = field(
        metadata = {"help" : 'The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".'}
    )
    max_tokens : int = field(
        metadata = {"help" : 'How many tokens the LLM is allowed to produce to classify each sample.'}
    )
    prompt : Optional[str] = field(
        default=None,
        metadata = {"help" : 'Optional prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.'}
    )
    prompt_role : Optional[str] = field(
        default='system',
        metadata = {"help" : 'What role to give the LLM prompt. Defaults to "system", meaning a system prompt. Can be replaced with "user" for models which do not work well with system prompts.'}
    )
    temperature : float = field(
        default=0,
        metadata = {"help" : 'Sampling temperature to be used. Higher = greater likelihood of low probability words.'}
    )
    top_p : Optional[float] = field(
        default=None,
        metadata = {"help" : 'If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if temperature > 0.'}
    )
    reasoning_effort : Optional[Literal["low","medium","high"]] = field(
        default=None,
        metadata = {"help" : "For cloud LLMs, controls the reasoning effort done by the models. For Anthropic models, reasoning efforts of 'low','medium','high' correlate to 1024,2048,4096 reasoning tokens respectively."}
    )
    claude_thinking_tokens : Optional[int] = field(
        default = None,
        metadata = {"help" : "For using Claude Sonnet 3.7 specifically - controls the number of tokens allowed for reasoning. Must be at least 1024. If set, temperature must be 1."}
    )
    out_path : str = field(
        default="results",
        metadata = {"help" : 'Which directory to save the evaluation result by default.'}
    )
    
    @classmethod
    def from_dict(cls, data_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})

    @classmethod
    def from_json(cls, json_file : str):
        with open(json_file) as f:
            data = json.load(f)
            f.close()
        return cls.from_dict(data) 

    def to_dict(self): return asdict(self)
    def save_json(self, path : str):
        data = self.to_dict()
        with open( path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
@dataclass
class EvaluationResult:
    """
    Raw LLM text classification evaluation results produced from ``evaluate.evaluate()``.

    Args:
        config (EvaluationConfig): The instructions given to the LLM to classify each sample.
        texts (list[str]): Every text sample in the evaluation dataset. (X_test)
        labels_pred (dict[str, list]): List of predicted class IDs (int) for each sample for each label. (y_pred)
        labels_true (dict[str, list]): List of true class IDs (int) for each sample for each label. (y_true)
        label_names (dict[str, list]): List of all class names for each label.
        llm_responses (list[ModelResponse]): LLM response for each sample.
        total_tokens (int): Number of input and output tokens used to perform the evaluation.
        total_time_elapsed (float): How long the evaluation took to run in seconds.
    """
    config : EvaluationConfig
    texts : list[str]
    labels_pred : dict[str, list]
    labels_true : dict[str, list]
    label_names : dict[str, list]
    llm_responses : list[ModelResponse]
    total_tokens : int
    total_time_elapsed : float

    @classmethod
    def from_dict(cls, data_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        result = cls(**{k: v for k, v in data_dict.items() if k in field_names})

        # We must parse EvaluationConfig from a dict
        result.config = EvaluationConfig.from_dict(result.config)
        
        llm_responses : list[ModelResponse] = []
        # We also must parse ModelResponse objects from dicts
        for response in result.llm_responses:
            llm_responses.append( ModelResponse.from_dict(response) )
        
        result.llm_responses = llm_responses

        return result
    
    @classmethod
    def from_json(cls, json_file : str):
        with open(json_file) as f:
            data = json.load(f)
            f.close()
        return cls.from_dict(data)

    def to_dict(self): return asdict(self)
    def save_json(self, path : str):
        data = self.to_dict()
        with open( path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_answers(self, incorrect_only : bool = False) -> pd.DataFrame:
        """
        Given raw LLM text classification evaluation data, return a DataFrame of the LLM's answers to each sample in human-readable format.

        Args:
            incorrect_only (bool, optional): Whether to only include the LLM's incorrect answers. Defaults to False.

        Returns:
            pd.DataFrame: A table containing each sample in the evaluation dataset, the LLM's response to each sample, and the predicted/actual labels.
        """

        y_pred, y_true = {}, {}

        for label, class_names in self.label_names.items():
            # Cast labels from int (class ID) -> str (class name)
            y_pred[label] = np.array([class_names[id] for id in self.labels_pred[label]])
            y_true[label] = np.array([class_names[id] for id in self.labels_true[label]])
        
        answers = {
        "Text" : np.array(self.texts),
        "LLM Response" : np.array([response.text for response in self.llm_responses]),
        "LLM Reasoning" : np.array([response.reasoning for response in self.llm_responses]),
        #"Predicted Label" : np.array(y_pred),
        #"True Label" : np.array(y_true),
        "Prediction Time" : np.array([response.latency for response in self.llm_responses]),
        "Total Tokens" : np.array([response.total_tokens for response in self.llm_responses])
        }

        for label in self.label_names.keys():
            answers[f"Predicted {label}"] = y_pred[label]
            answers[f"True {label}"] = y_true[label]

        answers = pd.DataFrame(answers)

        if incorrect_only:
            incorrect = False
            for label in self.label_names.keys():
                incorrect |= answers[f"Predicted {label}"] != answers[f"True {label}"]
            answers = answers[incorrect]

        return answers
    
    def get_time_elapsed(self) -> timedelta:
        """
        Return the total time elapsed running the evaluation.

        Returns:
            timedelta: Total time elapsed.
        """
        return timedelta(seconds=self.total_time_elapsed)
    
    def get_confusion_matrix(self, label_name : str, normalize : bool = True) -> np.ndarray:
        y_true, y_pred = self.labels_true[label_name], self.labels_pred[label_name]

        normalize = 'true' if normalize else None
        return confusion_matrix(y_true=y_true,y_pred=y_pred,normalize=normalize)
        
    def plot_confusion_matrix(self, label_name : str | None = None, char_limit:int = 15, max_classes:int = 15) -> ConfusionMatrixDisplay:
        """Generate a confusion matrix showing the prediction accuracy of the model for a given class label.

        Args:
            label_name (str, optional): What label to plot the confusion matrix for. Must be given if len(label_names) > 0. Defaults to None. 
            char_limit (int, optional): Truncate tick labels to this many characters. Defaults to 15.
            max_labels (int, optional): If there are more classes than this, hide all text in the graph altogether. Defaults to 15.

        Returns:
            ConfusionMatrixDisplay: The confusion matrix
        """
        
        if not label_name:
            if len(self.label_names) == 1: label_name = list(self.label_names.keys())[0]
            else: raise ValueError("label_name must be given if len(label_names) > 1")

        y_true, y_pred, label_names = self.labels_true[label_name], self.labels_pred[label_name], self.label_names[label_name]
        y_true = [label_names[i] for i in y_true]
        y_pred = [label_names[i] for i in y_pred]

        include_values = len(label_names) <= max_classes

        label_names_truncated = [f"{i[0:char_limit]}{"..." if len(i) > char_limit else ""}" for i in label_names]

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,y_pred=y_pred,labels=label_names,
            include_values=include_values, cmap=plt.cm.Blues,
            values_format = '.1f',
            xticks_rotation='vertical',
            normalize="true",
            display_labels=label_names_truncated
            )
            
        disp.ax_.set_title( f"{label_name} ({self.config.technique_name})" )

        if not include_values:
            disp.ax_.set_xticks([])
            disp.ax_.set_yticks([])

        return disp

    def _get_few_shot_examples(self,
                     incorrect_answers : DataFrame,
                     label_name : str,
                     true_class : str,
                     pred_class : str,
                     n : int = 1) -> DataFrame:
        """
        Helper function for ``get_few_shot_examples()``.
        Given a DataFrame of incorrect text classification LLM responses,
        return ``n`` rows from the label ``label_name`` where the
        true label was ``true_class`` and the predicted label was ``pred_class``.
    
        Args:
            incorrect_answers (DataFrame): Obtain with ``get_answers(incorrect_answers=True)``.
            label_name (str): Which label to use.
            true_class (str): Only retrieve rows with this as the true label.
            pred_class (str): Only retrieve rows with this as the predicted label.
            n (int): Number of rows to return.
    
        Returns:
            matches (DataFrame): Subset of rows from ``incorrect_answers`` which match the conditions.
        """
        true_class_column = "True " + label_name
        pred_class_column = "Predicted " + label_name
    
        inc = incorrect_answers # Shorthand
    
        matches = inc[(inc[true_class_column] == true_class) & (inc[pred_class_column] == pred_class)]
    
        matches = matches.sample(n=n)
        
        return matches
    
    def get_few_shot_examples(self,
        samples_per_class : int = 1,
        samples_per_true_label : int = 1,
        samples_per_pred_label : int = 1) -> str:
        """
        Generate few-shot examples for further text classification,
        prioritising classes that the LLM had the most difficulty classifying.

        Returns a maximum of ``num_classes * samples_per_class * samples_per_true_label * samples_per_pred_label`` examples,
        where ``num_classes`` is the number of unique classes in the dataset.
    
        Args:
            samples_per_class (int): How many examples to generate for each class (output label) in the dataset.
            samples_per_true_label (int): For each true label of each class, how many examples to generate.
            samples_per_pred_label (int): For each predicted label of each true label of each class, how many examples to generate.
    
        Returns:
            examples (str): Few-shot examples optimised to cover the LLM's weaknesses.
        """
    
        incorrect_answers = self.get_answers(incorrect_only=True)
        
        prompt = ""
        
        true_label_column_names = [i for i in incorrect_answers.columns.tolist() if "True" in i]
        
        # Get the column names for all classes in the incorrect answer dataframe
        true_label_column_names = [i for i in incorrect_answers.columns.tolist() if "True" in i]
        label_column_names = [i.lstrip("True").strip() for i in true_label_column_names]
        
        # For each class label:
        for label in label_column_names:
            #print('---------------------------')
            #print(label)
            #print('\n')
            true_label_column = "True " + label
            pred_label_column = "Predicted " + label
    
            incorrect = incorrect_answers[incorrect_answers[pred_label_column] != incorrect_answers[true_label_column]]
        
            # Get the most incorrectly predicted labels
            incorrect_labels = incorrect[true_label_column].value_counts()[:samples_per_class]
    
            # For each label name that was incorrectly predicted
            for true_label in incorrect_labels.keys().tolist():
                
                #print(f"True Label: {true_label}")
                
                # Get all answers which have that label as the true label
                pred_labels = incorrect[ incorrect[true_label_column] == true_label]
                pred_labels = pred_labels[pred_label_column].value_counts().keys()
                pred_labels = pred_labels[:samples_per_true_label]
                
                for pred_label in pred_labels:
                    #print(f"Pred Label: {pred_label}")
                    
                    examples = self._get_few_shot_examples(incorrect, label, true_label, pred_label, n=samples_per_pred_label)
    
                    for example in examples.to_dict(orient='records'):
                        prompt += "Question:\n"
                        prompt += example["Text"]
                        prompt += "\n\nAnswer:\n"
                        prompt += example[true_label_column]
                        prompt += "\n\n"
                        
        prompt += "Question:\n"
        return prompt

    def save(self, output_dir : str | None = None) -> None:
        """
        Creates human-readable results from raw LLM evaluation data.

        The following files are produced by this method:

        1. Confusion matrix for each label (``confusion_matrix_<label>.png``):
            - Graph visualisation of the LLM's accuracy for each class label.
        
        2. Classification report for each label (``evaluation_<label>.csv``):
            - Report of the LLM's accuracy, precision, recall, and F1 score for all classes for each class label.
        
        3. LLM answer data (``answers.csv, answers_incorrect.csv``):
            - A table containing all LLM responses and a table containing only the incorrect responses.

        4. Raw JSON data (``raw_output.json``):
            - Useful if you want to retrieve exact values from the output for future analysis.

        Args:
            output_dir (str, optional): Which folder to save the results into. Defaults to EvaluationConfig.out_path.
        """
        
        if not output_dir:
            output_dir = self.config.out_path

        # Make result name file safe
        result_path_name = _sanitize_string(self.config.technique_name)

        if not output_dir:
            output_dir = result_path_name
        else: output_dir = os.path.join( output_dir, result_path_name )

        # shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Dump the EvaluationResult data as a JSON file into "<output_dir>/raw_output.json"
        self.save_json( os.path.join(output_dir, "raw_output.json") )
        
        print(self.label_names)
        print(self.labels_pred)
        print(self.labels_true)

        # For each label:
        for label, class_names in self.label_names.items():
            
            label_name_sanitized = _sanitize_string(label)

            y_pred, y_true = self.labels_pred[label], self.labels_true[label]

            # Calculate accuracy, precision, recall, and F1 score
            classif_report = classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
            classif_report = pd.DataFrame(classif_report).transpose()

            classif_report.to_csv( os.path.join(output_dir, f"evaluation_{label_name_sanitized}.csv") )

            # Save the confusion matrix
            self.plot_confusion_matrix(label_name=label)
            plt.savefig( os.path.join(output_dir, f"confusion_matrix_{label_name_sanitized}.png"), dpi=200, bbox_inches='tight' )
            
        answers = self.get_answers(incorrect_only=False)
        incorrect_answers = self.get_answers(incorrect_only=True)

        answers.to_csv( os.path.join(output_dir, "answers.csv"), escapechar="\\" )
        incorrect_answers.to_csv( os.path.join(output_dir, "incorrect_answers.csv"), escapechar="\\" )

        plt.show()

def create_prompt(
    data_sample_name : str,
    label_names : dict[str, list[str]],
    chain_of_thought : bool = False,
    examples : str | None = None
    ) -> str:
    """
    Generate a text classification prompt using a given set of labels.
    This prompt can be used by a pre-trained language model to perform text classification.

    The prompt instructs the to answer in JSON format if there is more than one class to classify.
    You can optionally specify few-shot examples for the model to use and/or instruct the model to use Chain-of-Thought reasoning.

    Args:
        data_sample_name (str): A name to describe your data samples, e.g., "power outage report" or "serious injury report". Must be singular and in lowercase.
        label_names (dict[str, list]): A dict containing a list of label names for each class. The format is ``( class_name (str) : labels (list[str]) )``.
                                        If there is more than one class, the LLM will be instructed to provide an answer in JSON format.
        chain_of_thought (bool, optional): If true, instructs the LLM to answer with Chain-of-Thought reasoning rather than directly. Defaults to False.
        examples (str, optional): Few-shot examples to provide the LLM before classification. These examples are placed at the end of the prompt. Defaults to None.

    Returns:
        prompt (str): The text classification prompt.

    """
    
    prompt = f"You are an expert at classifying {data_sample_name}s"
    if len(label_names) == 1:

        label_names = label_names[list(label_names.keys())[0]]
        
        prompt += " into the following categories:\n\nCATEGORIES:\n"
        for label in label_names:
            prompt += f"- {label}\n"

        prompt += f"\nRead the following {data_sample_name} then answer with the name of the category which suits it best."

        if not chain_of_thought:
            prompt += f'\nAnswer with ONLY the name of the category, i.e., "{label_names[0]}".'
        else:
            prompt += f'\nWork out your answer step by step, then present your final answer using the name of the category, i.e., "{label_names[0]}".'

    else:
        units = ['','one','two','three','four','five','six','seven','eight','nine']
        
        prompt +=".\n\nYou are given two classification tasks."
        prompt +=f"\nYou should output the result as {units[len(label_names.keys())]} json fields as " + "{"
        
        for name in label_names.keys():
            prompt += f"{name} : {_sanitize_string(name)}_label, "
        
        prompt = prompt[:-2]
        prompt += "}\n"

        for name, items in label_names.items():
            prompt += f"\nFor {name}, given the {data_sample_name}, you are asked to classify it as one of the labels in the list "
            prompt += str(items)
            prompt += f" and change {_sanitize_string(name)}_label to the correct label in the list."

        if not chain_of_thought:
            prompt += f"\n\nOutput the {units[len(label_names.keys())]} json fields only and absolutely nothing else."
        else:
            prompt += f"\n\nWork out your answer step by step, then present your final answer using {units[len(label_names.keys())]} json fields."

    # Few-shot examples
    if examples:
        prompt += "\n\n"
        prompt += examples
    else:
        prompt += "\nNow it is your turn." if not chain_of_thought else "\nLet's think step by step."
    
    return prompt

def _get_class_id_from_model_response(model_response : str, label_names : list) -> int:
    """
    After getting an LLM to perform text classification,
    this function is used to extract class IDs from raw
    LLM outputs.

    Given a raw LLM output ``model_response``, attempt to find the name of
    a matching class label from ``label_names`` from the
    output and return its ID (position in the list).

    Assume that ``label_names`` contains a final
    "I don't know" entry for all examples that the LLM
    could not generate a class label for.

    The method to extract class IDs has 3 steps:
    1. Try to directly match the output with an item in ``label_names``.
    2. If ``model_response`` is an integer within the range of ``label_names``, return ``model_response`` casted as an int.
    3. If ``model_response`` only contains the first few letters of a class label, try to match it with a truncated class label.
       This allows you to run LLM evaluation with fewer max tokens than needed for the LLM to fully write each class label,
       which greatly reduces the time it takes for the model to classify each sample while retaining accuracy.
    4. Try to find the last matching class label name from the model's response using RegEx.
       This is useful for CoT prompting, where the model may end the answer with the class label it decided on.

    Args:
        model_response (str): The raw LLM output, ideally containing a class label.
        label_names (list): List of class label names with an additional final entry for unknown cases.

    Returns:
        class_id (int): The predicted class ID, or the final class ID if the LLM did not answer with any class label.
    """

    # Return a direct match if possible
    try:
        class_id = label_names.index(model_response)
        return class_id
    except Exception as e:
        pass
    
    model_response = model_response.lower().strip()
    
    # If the response is an integer, return it if it's a valid class ID
    if model_response.isdigit():
        try:
            class_id = int(model_response)
            # The integer must be within the range of class IDs
            if class_id >= 0 and class_id < len(label_names) - 2:
                return class_id
            else:
                # If it isn't, just return the last label ("I don't know").
                return len(label_names) - 1
        except Exception:
            pass

    # Match class labels if model_response is truncated.
    # E.g., "mean" -> "meanoftransportation" -> 5
    # This allows us to match labels even we don't
    # have enough tokens to write the entire name.
    # This allows us to optimise the evaluation
    # procedure by using lower max tokens.
    for i, label in enumerate(label_names):
        
        label = label.lower().strip()
        response_trimmed = model_response.replace(" ", "")
        label = label.replace(" ", "")
        label_truncated = label[0:len(response_trimmed)]

        if response_trimmed == label_truncated:
            return i
    
    # Regex match to last label in string
    try:
        # Regex escape each label name
        sanitised_label_names = [re.escape(label) for label in label_names]
        # Make every space character optional
        sanitised_label_names = [label.replace(" ", " *") for label in sanitised_label_names]
        # Concatenate all label names using boolean OR.
        match = "|".join(sanitised_label_names)

        # Find all instances of label name strings within the base string.
        matches = re.findall(match, model_response, flags = re.IGNORECASE)

        if matches:
            # Get the last matching label from the string.
            final_match = matches[-1]

            # Remove all capitalisation and whitespace to find the match's index
            # in the original list of label names.
            labels_sanitised = [i.lower().replace(" ", "") for i in label_names]
            match_sanitised = final_match.lower().replace(" ", "")

            # Return the matching class ID for the label.
            class_id = labels_sanitised.index(match_sanitised)
            return class_id
            
    except Exception:
        pass
    
    # If no class label is found in the LLM text, return the last label ("I don't know").
    return len(label_names) - 1

def _get_class_ids_from_model_response(model_response : str, label_names : dict) -> dict[str, int]:
    """
    After getting an LLM to perform text classification,
    this function is used to extract class IDs from raw
    LLM outputs using one or multiple class labels.

    If more than one class label is specified, this function
    assumes that the model response is a JSON string containing
    each class label as an entry.

    For each label in label_names, assume that the final entry
    is an "I don't know" label that is used if the model did not
    successfully select any label from the dataset.

    Args:
        model_response (str): The model's response. This must be in JSON format with an entry for each label if predicting from more than one label.
        label_names (dict): List of class names for each label with an additional final entry for unknown cases.
                            E.g., ``label_names["fruit"] = ["Apple", "Banana", "Orange", "Unknown"]``.

    Returns:
        class_ids (dict[str, int]): The predicted ID for each class.
                                    E.g., ``class_ids["fruit"] = 2``.
    """

    # If there is no model response, return all class IDs as unknown
    if not model_response:
        fail_dict = {}
        for label, class_names in label_names.items():
            fail_dict[label] = len(class_names) - 1
        return fail_dict

    if len(label_names) == 1:
        label = list(label_names.keys())[0] # Get name of first label
        label_names = list(label_names.values())[0] # Get all label values
        
        class_ids = {}
        class_ids[label] = _get_class_id_from_model_response(model_response, label_names)
        
        return class_ids

    # If we have more than one label, assume the model's response is in JSON format

    # Attempt to find a JSON object within the model's response
    match = re.search(r"{[^}]*}", model_response)
    if match: model_response = match.group()

    # Attempt to parse the model's response as a dictionary of predicted class labels
    try:
        response_dict = json.loads(model_response)
    except Exception:
        warnings.warn(f"Could not extract JSON from response: {model_response}")
        # If the model's response is not valid JSON, simply return each label as unknown
        class_ids = {}
        for label, class_names in label_names.items():
            class_ids[label] = len(class_names) - 1
        
        return class_ids
        #raise ValueError(f"Could not parse JSON data from model_response: {model_response}")
    
    # If the response JSON has as many entries as the labels:
    if len(response_dict) == len(label_names):

        # Fill in any missing label fields with
        # the item with a corresponding index
        # from the response JSON, assuming the
        # labels are in the correct order and
        # are misnamed.
        for i, label in enumerate(label_names.keys()):
            if label not in response_dict:
                try:
                    key = list(response_dict.keys())[i]
                    response_dict[label] = response_dict[key]
                except Exception: continue

    # For each class label
    class_ids = {}
    for label, class_names in label_names.items():

        # See if the model predicted a value for it
        try:
            pred_label_string = str(response_dict[label])
        # If not, return "unknown" for the class label
        except Exception:
            class_ids[label] = len(class_names) - 1
            continue
        
        # Extract the Class ID from the model's predicted label name
        class_id = _get_class_id_from_model_response(pred_label_string, class_names)
        class_ids[label] = class_id
    
    return class_ids

def evaluate(
    model : PretrainedLM,
    label_names : dict,
    eval_dataset : Dataset,
    eval_config : EvaluationConfig
    ) -> EvaluationResult:
    """
    Evaluate an LLM's text classification performance on a supervised dataset.

    Args:
        model (AutoModelForCausalLM): The LLM to use. It can be pre-trained or fine-tuned.
        tokenizer (AutoTokenizer): The tokenizer to use. This should come with the LLM.
        label_names (dict): List of all unique class names for each label.
                            E.g., ``label_names["fruit"] = ["Apple", "Banana", "Orange"]``.
        eval_dataset (Dataset): The evaluation dataset. Must be preprocessed (see ``finetune.preprocess_dataset()``).
        eval_config (EvaluationConfig): Defines what instructions and parameters to give to the LLM to classify each sample.

    Returns:
        EvaluationResult: Raw evaluation data, including all samples, predicted/actual labels, and the LLM's response for each sample.
    """
    
    # Add an "I don't know" label to the end of the label names list.
    # We will need this as a fallback if the LLM does not provide a
    # class label in its answer.
    label_names = label_names.copy() # Create a copy of label_names so we don't directly modify it
    for l in label_names.keys(): label_names[l].append("Unknown")

    labels_pred : list[dict[str,int]] = []
    llm_responses : list[ModelResponse] = []
    total_tokens : int = 0

    # Get all text inputs (X) in eval_dataset
    texts : list[str] = [message[0]['content'].strip() for message in eval_dataset['messages']]

    # Start logging how long the evaluation takes to run.
    time_started = time.time()
    
    # Pass in cloud LLM reasoning parameters as kwargs
    generation_kwargs = {}
    
    if eval_config.reasoning_effort is not None:
        generation_kwargs["reasoning_effort"] = eval_config.reasoning_effort

    if eval_config.claude_thinking_tokens is not None:
        generation_kwargs["thinking"] = {"type": "enabled", "budget_tokens": eval_config.claude_thinking_tokens}
    
    # For every sample:
    for text in tqdm(texts, "Evaluating model"):
        
        query_time = time.time()

        # Generate a classification prompt for the sample
        prompt = [
            {"role":eval_config.prompt_role, "content":eval_config.prompt},
            {"role":"user", "content":text}
        ]
        # Remove the system prompt from the chat template if none was specified
        if eval_config.prompt is None or eval_config.prompt == "":
            prompt.pop(0)

        # Get the LLM to generate an answer
        try:
            response = model.generate(
                prompt=prompt,
                max_new_tokens = eval_config.max_tokens,
                temperature=eval_config.temperature,
                top_p=eval_config.top_p,
                kwargs=generation_kwargs
            )

        # If something goes wrong, return an empty response
        except Exception as e:
            response = ModelResponse(
               text = "",
               prompt_tokens=0,
               completion_tokens=0,
               total_tokens=0,
               latency=time.time() - query_time,
               exception=e
            )
        
        total_tokens += response.total_tokens

        # Extract the class ID(s) from the LLM's answer if one exists
        pred_classes = _get_class_ids_from_model_response(response.text, label_names)

        labels_pred.append(pred_classes)
        llm_responses.append(response)

    total_time_elapsed = time.time() - time_started 

    # Get all class IDs from the label(s) in eval_dataset (y_true)
    groundtruth : list[str] = [message[-1]['content'] for message in eval_dataset['messages']]
    labels_true : list[dict[str, int]] = [_get_class_ids_from_model_response(label, label_names) for label in groundtruth]

    # Restructure labels from list of dicts to dict of lists
    labels_true = pd.DataFrame(labels_true).to_dict(orient='list')
    labels_pred = pd.DataFrame(labels_pred).to_dict(orient='list')

    return EvaluationResult(
        config=eval_config,
        texts=texts,
        labels_pred=labels_pred,
        labels_true=labels_true,
        label_names=label_names,
        llm_responses=llm_responses,
        total_tokens=total_tokens,
        total_time_elapsed=total_time_elapsed)
