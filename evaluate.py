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
class ClassificationMethod:
    """
    Defines how LLMs should be instructed to classify each text sample in a classification dataset.
    You can specify different configurations for different prompting techniques.

    NOTE: For each sample in the dataset, the LLM must output the *name* of the predicted class in its response.

    Args:
        name (str): The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned"
                    This name is used on the evaluation results.
        max_tokens (int): How many tokens the LLM is allowed to produce to classify each sample.
                          If you are planning on having your LLM output *just* the class label,
                          you can set this value to 1. The LLM will only return the first few
                          letters of the class label, but this is usually enough to identify
                          which label it selected.
        llm_instructions (str, optional): Optional system prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.
    """
    name : str
    max_tokens : int
    prompt : str | None = None
    # extractor_method : func

@dataclass
class EvaluationResult:
    """
    Raw LLM text classification evaluation results.

    Args:
        config (ClassificationMethod): 
        texts (list[str]): 
        labels_pred (list[int]): 
        labels_true (list[int]): 
        label_names (list[str]): 
        llm_responses (list[str]): 
    """
    config : ClassificationMethod
    texts : list[str]
    labels_pred : list[int]
    labels_true : list[int]
    label_names : list[str]
    llm_responses : list[str]

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
    2. If ``model_response`` only contains the first few letters of a class label, try to match it with a truncated class label.
       This allows you to run LLM evaluation with fewer max tokens than needed for the LLM to fully write each class label,
       which greatly reduces the time it takes for the model to classify each sample while retaining accuracy.
    3. Try to find the last matching class label name from the model's response using RegEx.
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
    
    # Match class labels if model_response is truncated.
    # E.g., "mean" -> "meanoftransportation" -> 5
    # This allows us to match labels even we don't
    # have enough tokens to write the entire name.
    # This allows us to optimise the evaluation
    # procedure by using lower max tokens.
    for i, label in enumerate(label_names):
        
        label_truncated = label.lower().replace(" ", "")[0:len(model_response)]
        if model_response == label_truncated:
            return i
    
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

    else:
        # If no class label is found in the LLM text, return the last label.
        return len(label_names) - 1

def evaluate(
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    label_names : list,
    eval_dataset : Dataset,
    eval_config : ClassificationMethod
    ) -> EvaluationResult:
    """
    Evaluate an LLM's text classification performance on a supervised dataset.

    Args:
        model (AutoModelForCausalLM): The LLM to use. It can be pre-trained or fine-tuned.
        tokenizer (AutoTokenizer): The tokenizer to use. This should come with the LLM.
        label_names (list): The name of each class label in the evaluation dataset.
        eval_dataset (Dataset): The evaluation dataset. Must be preprocessed (see ``finetune.preprocess_dataset()``).
        eval_config (ClassificationMethod): Controls what instructions to give to the LLM to classify each sample.

    Returns:
        EvaluationResult: Raw evaluation data, including all samples, predicted/actual labels, and the LLM's response for each sample.
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
        pred_class = _get_class_id_from_model_response(response, label_names)

        labels_pred.append(pred_class)
        llm_responses.append(response)

    # Get all class label IDs (y_true) in eval_dataset
    groundtruth = [message[-1]['content'] for message in eval_dataset['messages']]
    labels_true = [_get_class_id_from_model_response(label, label_names) for label in groundtruth]

    return EvaluationResult(
        config=config,
        texts=texts,
        labels_pred=labels_pred,
        labels_true=labels_true,
        label_names=label_names,
        llm_responses=llm_responses)

def get_answers(result : EvaluationResult, incorrect_only : bool = False) -> pd.DataFrame:
    """
    Given raw LLM text classification evaluation data, return a DataFrame of the LLM's answers to each sample in human-readable format.

    Args:
        result (EvaluationResult): The raw LLM evaluation data produced from ``evaluate()``.
        incorrect_only (bool, optional): Whether to only include the LLM's incorrect answers. Defaults to False.

    Returns:
        pd.DataFrame: A table containing each sample in the evaluation dataset, the LLM's response to each sample, and the predicted/actual labels.
    """
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
    

def save_evaluation_result(result : EvaluationResult, output_dir : str = os.path.join("output", result.config.name)) -> None:
    """
    Creates human-readable results from raw LLM evaluation data.

    The following files are produced by this method:

    1. Confusion matrix (``confusion_matrix.png``):
       Graph visualisation of the LLM's accuracy.
       
    2. Classification report (``evaluation.csv``):
       Report of the LLM's accuracy, precision, recall, and F1 score for all classes.
       
    3. LLM answer data (``answers.csv, answers_incorrect.csv``):
       A table containing all LLM responses and a table containing only the incorrect responses.

    Args:
        result (EvaluationResult): The raw LLM evaluation data produced from ``evaluate()``.
        output_dir (str): Which folder to save the results in. Defaults to ``output/<configuration_name>``.
    """
    # shutil.rmtree(output_dir)
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
    