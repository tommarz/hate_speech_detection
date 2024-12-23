from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_and_freeze_model(model_name: str, tokenizer_name: str = None):
    """
    Load a Hugging Face model and tokenizer, freeze all weights except for the classifier module.

    Args:
        model_name (str): Name of the model to load from Hugging Face.
        tokenizer_name (str, optional): Name of the tokenizer to load. Defaults to the same as `model_name`.

    Returns:
        model: Hugging Face model with frozen weights except for the classifier module.
        tokenizer: Hugging Face tokenizer.
    """
    # Default tokenizer to the same as model name if not specified
    if tokenizer_name is None:
        tokenizer_name = model_name

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Freeze all layers except for the classifier module
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    return tokenizer, model