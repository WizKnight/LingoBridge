from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_data(dataset_name="alek1001/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus-text", 
                    source_lang="en", 
                    target_lang="ja", 
                    max_length=128):
    """
    Preprocesses the translation dataset.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        DatasetDict: The preprocessed dataset.
    """

    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Model Name
    model_name = "Helsinki-NLP/opus-mt-en-jap"

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocessing function
    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples]
        targets = [ex[target_lang] for ex in examples]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the preprocessing function to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Remove unnecessary columns
    tokenized_dataset = tokenized_dataset.remove_columns(["id", source_lang, target_lang])

    return tokenized_dataset

if __name__ == "__main__":
    preprocessed_dataset = preprocess_data()
    print(preprocessed_dataset)