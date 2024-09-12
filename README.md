# Quantifying Decision Support Level of Explainable Automatic Classification of Diagnoses in Spanish Medical Records
This repository contains the code used for the scientific paper titled "Quantifying Decision Support Level of 
Explainable Automatic Classification of Diagnoses in Spanish Medical Records." The code includes scripts for 
model training (Train_transformer.py) and explainability extraction and evaluation (Get_transformer_explainability.py). The usage of this software is open for research purposes and is bound to the citation of the article:
Nuria Lebeña, Alicia Pérez, Arantza Casillas,
Quantifying decision support level of explainable automatic classification of diagnoses in Spanish medical records,
Computers in Biology and Medicine,
Volume 182,
2024,
109127,
ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2024.109127.

#### Installation
Before running the scripts, ensure that you have all the required libraries installed. The necessary libraries are 
listed in the requirements.txt file. Python version 3.9.7 is recommended. You can install them using the following 
command:

```pip install -r requirements.txt```

## Training and Evaluation Script (Train_transformer.py)
This script is designed to train and evaluate a multi-label classification clinical-longformer for diagnosing medical 
conditions from Spanish medical records. It allows for fine-tuning and evaluation on custom datasets.

```
python Train_transformer.py --training_data_dir <path_to_training_data> --test_data_dir <path_to_test_data> 
--eval_data_dir <path_to_eval_data> --mode <train_or_test> [other_arguments]

```

### Arguments

--training_data_dir: Directory containing the training dataset.

--test_data_dir: Directory containing the test dataset.

--eval_data_dir: Directory containing the evaluation dataset.

--mode: Mode of operation, either "train" or "test".

--labels_ids: Path to the file containing label IDs.

--labels_default_start_index: Default start index for the labels (default: None).

--csv_sep: Separator used in the CSV files (default: "\t").

--model_id: Pretrained model name (default: "allenai/longformer-base-4096").

--model_type: Pretrained model type (default: "longformer").

--model_path: Directory containing the model to load (default: "model_id").

--model_out: Directory to save the model (default: "output").

--output_dir: Directory to save the predictions (default: "output").

--experiment_name: Name of the experiment (default: "longformer").

--output_attentions: Whether to output attentions (default: False).

--no_cache: Whether to cache the features (default: False).

--max_len: Maximum length of the sequence (default: 512).

--batch_size: Batch size for training (default: 4).

--epochs: Number of training epochs (default: 50).

--lr: Learning rate (default: 1e-5).

--warmup_proportion: Proportion of training to perform linear learning rate warmup for (default: 0.1).

--warmup_steps: Number of warmup steps (default: 125).

--grad_accum: Number of gradient accumulation steps (default: 1).

--adam_epsilon: Epsilon for Adam optimizer (default: 1e-8).

--weight_decay: Weight decay (default: 0.07).

--max_grad_norm: Maximum gradient norm (default: 1.0).

--metric_for_best_model: Metric for the best model (default: "f1").

--seed: Random seed for initialization (default: 42).

### Functionality
Data Loading: The script initializes a DataHolder object to load the training, development, and test datasets, as well
as the label mappings.

Model Initialization: Depending on the mode (train/test), the script initializes a pretrained model (Longformer in this 
case) from the specified model path.

Training Mode: If the mode is set to "train", the script sets up the training arguments, optimizer, and scheduler. It 
then trains the model using the Trainer class from the Hugging Face Transformers library.

Evaluation Mode: If the mode is set to "test", the script evaluates the model on the test dataset and saves the 
predictions and evaluation metrics to the specified output directory.
### Output
Model Checkpoints: During training, the best model checkpoint is saved based on the specified evaluation metric.

Predictions: The script saves both binary and raw predictions in CSV format.

Evaluation Metrics: The script saves the evaluation metrics to a text file.


## Script 2: Explainability Script (Get_transformer_explainability.py)
This script is designed to provide explainability for the predictions made by the transformer model. It uses various
explainability methods to highlight important features contributing to the model's decisions.

Usage
You can run the script with the following command:
```
python Get_transformer_explainability.py --test_data_dir <path_to_test_data> [other_arguments]
```

### Arguments
--test_data_dir: Directory containing the test dataset.

--labels_default_start_index: Default start index for the labels (default: None).

--text_column_name: Name of the text column in the CSV file (default: "text").

--csv_sep: Separator used in the CSV files (default: "\t").

--model_id: Pretrained model name (default: "allenai/longformer-base-4096").

--model_type: Pretrained model type (default: "longformer").

--model_path: Directory containing the model to load (default: "model_id").

--from_file: Whether to load the model from the specified model path (default: False).

--json_path: Path to the JSON file containing the model (used if --from_file is specified).

--language: Language of the texts (default: "english").

--explainability_method: Explainability method to use (options: "lime", "shap", "ig", "all"; default: "all").

--n_highlighted_words: Number of top words to highlight in the explainability results (default: 5).

--no_stopwords: Whether to remove stopwords in the explainability results (default: False).

--n_grams: Whether to use n-grams in the explainability results (default: False).

--n_grams_size: Size of the n-grams to use (default: 3).

--n_top_labels: Number of top labels to return in the explainability results (default: 4).

--output_dir: Directory to save the generated files (default: "output").

--experiment_name: Name of the experiment (default: "longformer").

--no_return_html: Whether to not return an HTML file per instance with the explainability results (default: False).

--no_return_metrics: Whether to not return a metrics file with the explainability results (default: False).

--html_graph_type: Type of graph to return in the HTML file (options: "both", "general", "separated by label"; default: "both").

### Functionality
Data Loading: The script loads the test data, including texts and labels.

Model Loading: Depending on the specified arguments, the script either loads a pretrained model directly or from a
provided file.

Explainability Methods: The script supports multiple explainability methods such as LIME, SHAP, and IG. It generates
explainability results for each method and saves them in the specified output directory.

Output: The script generates JSON files with the explainability results and their evaluation using Leberage metric,
optionally HTML files for visualization, 
and metrics files if specified.

### Output
Explainability Results: JSON files containing the top words or n-grams and their weights for each document.

HTML Files: Optional HTML files for visualizing explainability results.

Metrics: Optional metrics files evaluating the explainability results.
