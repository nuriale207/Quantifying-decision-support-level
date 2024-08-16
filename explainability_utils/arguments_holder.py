import os

import pandas as pd

from explainability_utils.integrated_gradients import generate_colored_text, get_top_n_attributions_dict
from utils.model_loader import load_from_pretrained_model_tokenizer, get_model_embeddings


class ExplainabilityArgumentsHolder:

