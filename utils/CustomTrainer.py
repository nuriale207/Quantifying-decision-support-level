from torch import nn
from transformers import Trainer
import torch

from utils.model_loader import binarize_predictions


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     if "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #
    #     other_model_outputs = self.other_model(**inputs)  # Compute predictions from the other model
    #
    #     # Compute the loss using your custom loss function
    #     loss = self.custom_loss_function(model(**inputs), labels, other_model_outputs)
    #
    #     return (loss, other_model_outputs) if return_outputs else loss

    def add_model_for_inference(self, model):
        self.hierarchial_model = model
        self.hierarchial_model.to(self.args.device)


    def add_inputs_for_inference(self, main_inputs,chapter_inputs):
        self.main_inputs = inputs
        self.chapter_inputs = inputs
    def compare_predictions(self, probabilities,probabilities_prev):
        pass