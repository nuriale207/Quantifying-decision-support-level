import sys

from explainability_utils.Explainable_model import ExplainableModel

sys.path.append("..")
from transformers import pipeline

from lime.lime_text import LimeTextExplainer
import numpy as np


#Class lime inherits from the Explainable model class
class Lime(ExplainableModel):
    ## Define the constructor
    def __init__(self, model, tokenizer,text,top=5):
        super().__init__(model, tokenizer, text)
        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        self.labels_names = list(self.classifier.model.config.id2label.values())
        self.explainer = LimeTextExplainer(class_names=self.labels_names)
        self.top_n=top


    def predict_proba(self,texts):
        predictions = self.classifier(texts)
        predictions_vector = []
        for pred in predictions:
            predictions_vector.append([prediction['score'] for prediction in pred])

        # convert to numpy array
        predictions_vector = np.array(predictions_vector)
        return predictions_vector

    #override get by word weights
    #this methods is overriden. Override it to make it work with lime

    def get_by_word_weights(self):
        exp = self.explainer.explain_instance(self.text, self.predict_proba, num_features=self.top_n, top_labels=4)
        #Put each class explanation in a dictionary
        class_explanations = {}
        for label_idx in exp.available_labels():
            class_explanations[self.labels_names[label_idx]] =dict(exp.as_list(label=label_idx))

        #Create the word list
        word_list=exp.domain_mapper.indexed_string.inverse_vocab
        print(word_list)
        print(class_explanations)
        return word_list, class_explanations


