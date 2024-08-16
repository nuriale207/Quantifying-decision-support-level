import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


def load_from_pretrained_model_tokenizer(model_id, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def get_model_embeddings(model, model_type):
    if model_type == "longformer":
        model_embeddings = model.longformer.embeddings
    elif model_type=="bert":
        model_embeddings=model.bert.embeddings
    return model_embeddings

def get_prediction(text,model, tokenizer, max_length=512):

    """
    Get the prediction from the model
    :param text: text to predict
    :param model: model
    :param tokenizer: tokenizer
    :param max_length: max length
    :return: the prediction
    """
    # Encode text to get text_ids
    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    # Get the input ids and tokens list
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # get the model output
    input_ids=torch.tensor([input_ids], device='cuda')
    input_ids.to('cuda')
    output = model(input_ids)

    # Apply softmax to get the probabilities
    #probabilities = torch.nn.functional.softmax(output[0], dim=-1)
    # probabilities=torch.nn.functional.sigmoid(output[0])
    probabilities = torch.sigmoid(output[0])
    return probabilities

def binarize_predictions(predictions, top_n=4,top=False,threshold=0.5):
    """
    Binarize the predictions based on the threshold

    Args:
        predictions (torch.tensor): The predictions from the model
        threshold (float, optional): The threshold to binarize the predictions. Defaults to 0.5.

    Returns:
        torch.tensor: The binarized predictions
    """
    if top:
        top_n_predictions = torch.topk(predictions, top_n, dim=1)
        binary_predictions = torch.zeros(predictions.shape)
        binary_predictions[0][top_n_predictions.indices] = 1
        return binary_predictions
    else:
        binary_predictions = torch.zeros(predictions.shape)
        binary_predictions[predictions >= threshold] = 1
        binary_predictions[predictions < threshold] = 0
        return binary_predictions