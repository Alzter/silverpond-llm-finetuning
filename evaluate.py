from dataclasses import dataclass
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, shutil
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

@dataclass
class EvaluationConfig:
    name : str
    max_tokens : int
    llm_instructions : str
    # extractor_method : func

@dataclass
class EvaluationResult:
    config : EvaluationConfig
    texts : list[str]
    labels_pred : list[int]
    labels_true : list[int]
    label_names : list[str]
    llm_responses : list[str]

def _get_class_id_from_string(string, label_names : list) -> int:
    """_summary_

    Args:
        string (_type_): _description_
        label_names (list): _description_

    Returns:
        int: _description_
    """

    # Return a direct match if possible
    try:
        class_id = label_names.index(string)
        return class_id
    except Exception as e:
        pass

    string = string.lower().strip()
    
    # Match class labels if string is truncated.
    # E.g., "mean" -> "meanoftransportation" -> 5
    # This allows us to match labels even we don't
    # have enough tokens to write the entire name.
    # This allows us to optimise the evaluation
    # procedure by using lower max tokens.
    for i, label in enumerate(label_names):
        
        label_truncated = label.lower().replace(" ", "")[0:len(string)]
        if string == label_truncated:
            return i
    
    # Concatenate all label names using boolean OR.
    match = "|".join(label_names).lower().replace(" ", r"\s*")

    # Find all instances of label name strings within the base string.
    matches = re.findall(match, string)

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

    else:
        # If no class label is found in the LLM text, return the last label.
        return len(label_names) - 1

def evaluate(
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    label_names : list,
    eval_dataset : Dataset,
    eval_config : EvaluationConfig
    ) -> EvaluationResult:
    """_summary_

    Args:
        model (AutoModelForCausalLM): _description_
        tokenizer (AutoTokenizer): _description_
        label_names (list): _description_
        eval_dataset (Dataset): _description_
        eval_config (EvaluationConfig): _description_

    Returns:
        EvaluationResult: _description_
    """

    # Add an "I don't know" label to the end of the label names list.
    # We will need this as a fallback if the LLM does not provide a
    # class label in its answer.
    label_names.append("Unknown")

    labels_pred = []
    llm_responses = []

    # Get all text inputs (X) in eval_dataset
    texts = [message[0]['content'].strip() for message in eval_dataset['messages']]

    # For every sample:
    for text in tqdm(texts, "Evaluating model"):
        
        # Generate a classification prompt for the sample
        prompt = [
            {"role":"system", "content":eval_config.llm_instructions},
            {"role":"user", "content":text}
        ]
        # Remove the system prompt from the chat template if none was specified
        if eval_config.llm_instructions is None or eval_config.llm_instructions == "":
            prompt.pop(0)

        # Get the LLM to generate an answer
        response = ft.generate(
                            prompt=prompt, model=model, tokenizer=tokenizer,
                            max_new_tokens = eval_config.max_tokens,
                            skip_special_tokens=True, response_only=True,
                            do_sample=True, temperature=0.1
                            )
        
        # Extract the class ID from the LLM's answer if one exists
        pred_class = _get_class_id_from_string(response, label_names)

        labels_pred.append(pred_class)
        llm_responses.append(response)

    # Get all class label IDs (y_true) in eval_dataset
    groundtruth = [message[-1]['content'] for message in eval_dataset['messages']]
    labels_true = [_get_class_id_from_string(label, label_names) for label in groundtruth]

    return EvaluationResult(
        config=config,
        texts=texts,
        labels_pred=labels_pred,
        labels_true=labels_true,
        label_names=label_names,
        llm_responses=llm_responses)

def get_answers(result : EvaluationResult, incorrect_only : bool = False) -> pd.DataFrame:

    # Cast labels from int (class ID) -> str (class name)
    y_pred = [label_names[id] for id in result.y_pred]
    y_true = [label_names[id] for id in result.y_true]
    
    answers = {
      "Text" : np.array(result.texts)[mask],
      "Predicted Label" : np.array(y_pred),
      "True Label" : np.array(y_true),
      "LLM Response" : np.array(result.llm_responses)
    }
    
    answers = pd.DataFrame(answers, index=index)

    if incorrect_only: answers = answers.loc[answers['Predicted Class'] != answers['True Class']]

    return answers
    

def save_evaluation_result(result : EvaluationResult, output_dir : str) -> None:
    """_summary_

    Args:
        result (EvaluationResult): _description_
        output_dir (str): _description_
    """
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate accuracy, precision, recall, and F1 score
    classif_report = classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
    classif_report = pd.DataFrame(classif_report).transpose()

    classif_report.to_csv( os.path.join(output_dir, "evaluation.csv"), index=False )

    y_pred, y_true, label_names = result.y_pred, result.y_true, result.label_names

    label_names = label_names[0:len(np.unique(y_pred))]

    cm = confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true')

    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(
        cmap = plt.cm.Blues,
        xticks_rotation='vertical',
        text_kw={'fontsize': 6},
        values_format='.0%'
    )

    disp.ax_.set_title( result.config.name )

    plt.savefig( os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight' )

    answers = get_answers(result, incorrect_only=False)
    answers.to_csv( os.path.join(output_dir, "answers.csv"), escapechar="\\" )

    incorrect_answers = get_answers(result, incorrect_only=True)
    incorrect_answers.to_csv( os.path.join(output_dir, "incorrect_answers.csv"), escapechar="\\" )
    