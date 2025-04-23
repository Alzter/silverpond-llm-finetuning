import dataclasses
from dataclasses import dataclass, asdict
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, shutil, re
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
import finetune as ft
import pandas as pd
import numpy as np
import time, json
from datetime import timedelta

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
    Defines how LLMs should be instructed to classify each text sample in a classification dataset.
    You can specify different configurations for different prompting techniques.

    NOTE: For each sample in the dataset, the LLM must output the *name* of the predicted class in its response.

    Args:
        name (str): The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".
        max_tokens (int): How many tokens the LLM is allowed to produce to classify each sample.
                          If you are planning on having your LLM output *just* the class label,
                          you can set this value to 1. The LLM will only return the first few
                          letters of the class label, but this is usually enough to identify
                          which label it selected. See ``_get_class_id_from_model_response()`` for implementation details.
        prompt (str, optional): Optional system prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.
    """
    name : str
    max_tokens : int
    prompt : str | None = None
    # extractor_method : func
    
    @classmethod
    def from_dict(cls, data_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})

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
        llm_responses (list[str]): Raw LLM response to each sample.
        prediction_times (list[float]): How long it took the LLM to classify each sample in seconds.
        total_time_elapsed (float): How long the evaluation took to run overall in seconds.
    """
    config : EvaluationConfig
    texts : list[str]
    labels_pred : dict[str, list]
    labels_true : dict[str, list]
    label_names : dict[str, list]
    llm_responses : list[str]
    prediction_times : list[float]
    total_time_elapsed : float

    @classmethod
    def from_dict(cls, data_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        result = cls(**{k: v for k, v in data_dict.items() if k in field_names})
        result.config = EvaluationConfig.from_dict(result.config)
        return result
    
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
        "LLM Response" : np.array(self.llm_responses),
        #"Predicted Label" : np.array(y_pred),
        #"True Label" : np.array(y_true),
        "Prediction Time" : np.array(self.prediction_times)
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
        
    def plot_confusion_matrix(self, label_name : str, char_limit:int = 15, max_classes:int = 15) -> ConfusionMatrixDisplay:
        """Generate a confusion matrix showing the prediction accuracy of the model for a given class label.

        Args:
            label_name (str): What label to plot the confusion matrix for.
            char_limit (int, optional): Truncate tick labels to this many characters. Defaults to 15.
            max_labels (int, optional): If there are more classes than this, hide all text in the graph altogether. Defaults to 15.

        Returns:
            ConfusionMatrixDisplay: The confusion matrix
        """
        y_true, y_pred, label_names = self.labels_true[label_name], self.labels_pred[label_name], self.label_names[label_name]
        y_true = [label_names[i] for i in y_true]
        y_pred = [label_names[i] for i in y_pred]

        include_values = len(label_names) <= max_classes

        label_names_truncated = [f"{i[0:char_limit]}{"..." if len(i) > char_limit else ""}" for i in label_names]

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,y_pred=y_pred,labels=label_names,
            include_values=include_values, cmap=plt.cm.Blues,
            xticks_rotation='vertical',
            normalize="true",
            display_labels=label_names_truncated
            )
            
        disp.ax_.set_title( f"{label_name} ({self.config.name})" )

        if not include_values:
            disp.ax_.set_xticks([])
            disp.ax_.set_yticks([])

        return disp

    def save(self, output_dir : str = "results") -> None:
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
            output_dir (str, optional): Which folder to save the results into. Defaults to "results".
        """
        
        # Make result name file safe
        result_path_name = _sanitize_string(self.config.name)

        if not output_dir:
            output_dir = result_path_name
        else: output_dir = os.path.join( output_dir, result_path_name )

        # shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Dump the EvaluationResult data as a JSON file into "<output_dir>/raw_output.json"
        self.save_json( os.path.join(output_dir, "raw_output.json") )

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
    
    try:
        # Concatenate all label names using boolean OR.
        match = "|".join(label_names).lower().replace(" ", r"\s*")

        # Find all instances of label name strings within the base string.
        matches = re.findall(match, model_response)

        # If the string contains at least one instance of a class label:
        if len(matches) > 0:
            # Get the last matching label from the string.
            final_match = matches[-1]

            # Remove all capitalisation, non-alphabetic characters, and whitespace
            labels_sanitised = [re.sub("[^a-z]", "", label.lower()) for label in label_names]
            match_sanitised = re.sub("[^a-z]", "", final_match.lower())
            
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

    if len(label_names) == 1:
        label_names = list(label_names.values())[0]
        return _get_class_id_from_model_response(model_response, label_names)
    
    # If we have more than one label, assume the model's response is in JSON format

    # Attempt to parse the model's response as a dictionary of predicted class labels
    try:
        response_dict = json.loads(model_response)
    except Exception:
        
        print(f"Could not extract JSON from response: {model_response}")
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
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
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
        eval_config (EvaluationConfig): Controls what instructions to give to the LLM to classify each sample.

    Returns:
        EvaluationResult: Raw evaluation data, including all samples, predicted/actual labels, and the LLM's response for each sample.
    """
    
    # Add an "I don't know" label to the end of the label names list.
    # We will need this as a fallback if the LLM does not provide a
    # class label in its answer.
    for l in label_names.keys(): label_names[l].append("Unknown")

    labels_pred = []
    llm_responses = []

    # Get all text inputs (X) in eval_dataset
    texts = [message[0]['content'].strip() for message in eval_dataset['messages']]

    # Start logging how long the evaluation takes to run.
    time_elapsed = [time.time()]

    # For every sample:
    for text in tqdm(texts, "Evaluating model"):
        
        # Generate a classification prompt for the sample
        prompt = [
            {"role":"system", "content":eval_config.prompt},
            {"role":"user", "content":text}
        ]
        # Remove the system prompt from the chat template if none was specified
        if eval_config.prompt is None or eval_config.prompt == "":
            prompt.pop(0)

        # Get the LLM to generate an answer
        response = ft.generate(
                            prompt=prompt, model=model, tokenizer=tokenizer,
                            max_new_tokens = eval_config.max_tokens
                            )
        
        # Extract the class ID(s) from the LLM's answer if one exists
        pred_classes = _get_class_ids_from_model_response(response, label_names)

        labels_pred.append(pred_classes)
        llm_responses.append(response)

        # Add each iteration to the time taken.
        time_elapsed.append(time.time())

    total_time_elapsed = time_elapsed[-1] - time_elapsed[0]
    prediction_times = list(np.diff(time_elapsed))

    # Get all class IDs from the label(s) in eval_dataset (y_true)
    groundtruth = [message[-1]['content'] for message in eval_dataset['messages']]
    labels_true = [_get_class_ids_from_model_response(label, label_names) for label in groundtruth]

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
        prediction_times=prediction_times,
        total_time_elapsed=total_time_elapsed)
