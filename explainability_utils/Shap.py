from transformers import pipeline
# Load the model
import sys
from itertools import groupby

sys.path.append("..")

import shap

shap.initjs()

from utils.model_loader import load_from_pretrained_model_tokenizer
from utils.data_handler import DataHandler

import torch

import re

from transformers import pipeline
from explainability_utils.Explainable_model import ExplainableModel


#Class shap inherits from the Explainable model class
class Shap (ExplainableModel):
    ## Define the constructor
    def __init__(self, model, tokenizer,text):
        super().__init__(model, tokenizer, text)

        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)

        self.classifier.function_to_apply = "sigmoid"

        self.explainer = shap.Explainer(self.classifier)

        self.compute_token_weights()


    # def get_token_weights(self):
    #     return self.token_weights

    def compute_token_weights(self):

        self.token_weights = {}

        max_token_len=1512
        tokens=self.tokenizer.tokenize(self.text[0])
        if len(tokens)>max_token_len:
            tokens=tokens[:max_token_len]

        #Select just the text that can be processed by the model
        self.text[0]=self.tokenizer.convert_tokens_to_string(tokens)

        self.shap_values = self.explainer(self.text)

        self.labels_names =self.shap_values.output_names

        self.token_list=self.shap_values.data[0][1:-1]

        #Create token_ids2words
        text_ids = self.tokenizer.encode(self.text[0], truncation=True, add_special_tokens=False)

        encoded = self.tokenizer(self.text[0], truncation=True, add_special_tokens=False, return_tensors="pt")
        indexes_list = list(range(len(text_ids)))
        words2token_ids_dict = dict(zip(indexes_list, encoded.word_ids()))

        grouped_keys = {k: [key for key, _ in v] for k, v in
                        groupby(words2token_ids_dict.items(), key=lambda item: item[1])}

        # Create the final list
        self.token_ids2word = list(grouped_keys.values())

        label_indexes=self.get_top_labels_indexes()
        i=0
        for i in label_indexes:
            label=self.labels_names[i]
            j=0
            token_label_shaps=[]
            for j in range(len(self.shap_values.values[0,:,i])):
                token_label_shaps.append((self.shap_values.data[0][j],self.shap_values.values[0,:,i][j]))
                j+=1
            token_label_shaps=token_label_shaps[1:-1]
            self.token_weights[label]=token_label_shaps
            i+=1


        return self.token_weights

    def get_top_labels_indexes(self):

        # self.shap_values = self.explainer(self.text)

        classifier_output = self.classifier(self.text)

        predictions_dict= {}
        suma=0
        for prediction in classifier_output[0]:
            predictions_dict[prediction["label"]]=float(prediction["score"])

        predictions_dict=dict(sorted(predictions_dict.items(), key=lambda x: x[1], reverse=True))

        #Check if there are any labels with score higher than 0.5
        if all(value < 0.5 for value in predictions_dict.values()):
            #get the top 4 labels
            top_labels_ids = list(predictions_dict.keys())[:4]
        else:
            #Get the labels with score higher than 0.5
            top_labels_ids = [k for k, v in predictions_dict.items() if v > 0.5]

        #get the top 4 labels indexes in the shap_values
        top_labels_indexes = []
        for label in top_labels_ids:
            act_index = self.shap_values.output_names.index(label)
            top_labels_indexes.append(act_index)
        return top_labels_indexes

    # def get_by_word_weights(self):
    #     """
    #             """
    #
    #     self.word_shaps = {}
    #     for key in self.token_weights.keys():
    #         label_token_shaps=self.token_weights[key]
    #         word_shaps = []
    #         for ls in self.token_ids2word:
    #             attributions = 0.0
    #             for index in ls:
    #                 attributions += label_token_shaps[index][1]
    #             word_shaps.append(attributions)
    #
    #         reconstructed_text, words_list = self.get_reconstructed_text()
    #         label_word_shaps = list(zip(words_list, word_shaps))
    #         self.word_shaps[key] = label_word_shaps
    #
    #     # Average the attributions of repeated words
    #     by_label_word_shap_dict = {}
    #     for key in self.word_shaps.keys():
    #         label_word_shaps = self.word_shaps[key]
    #         word_shaps_dict = {}
    #         for word, attribution in label_word_shaps:
    #             if word in word_shaps_dict.keys():
    #                 word_shaps_dict[word].append(attribution)
    #             else:
    #                 word_shaps_dict[word] = [attribution]
    #         for word in word_shaps_dict.keys():
    #             word_shaps_dict[word] = sum(word_shaps_dict[word]) / len(word_shaps_dict[word])
    #         by_label_word_shap_dict[key] = word_shaps_dict
    #
    #     # Update the word attributions list of lists
    #
    #     for key in self.word_shaps.keys():
    #         label_word_shaps = self.word_shaps[key]
    #         word_shaps_dict = by_label_word_shap_dict[key]
    #         for i in range(len(label_word_shaps)):
    #             word = label_word_shaps[i]
    #             attribution = word_shaps_dict[word[0]]
    #             label_word_shaps[i] = (word[0], attribution)
    #
    #     return self.word_shaps,by_label_word_shap_dict

    # def get_reconstructed_text(self):
    #     """
    #     Given the list of tokens and the list of list with the tokens that correspond to each word, return the reconstructed
    #     text
    #     :param token_list: list with the tokens
    #     :param token_ids2word: list of list with the tokens that correspond to each word
    #     :return: reconstructed text
    #     """
    #     reconstructed_text = ""
    #     for list in self.token_ids2word:
    #         for index in list:
    #             token=self.token_list[index].replace(" ","")
    #             reconstructed_text += token
    #         reconstructed_text += " "
    #     reconstructed_text = reconstructed_text.replace("Ġ", "")
    #     reconstructed_text = reconstructed_text.replace('Ċ', "")
    #
    #     words_list = reconstructed_text.split(" ")
    #
    #     return reconstructed_text, words_list