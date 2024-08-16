from explainability_utils.Explainable_model import ExplainableModel
from explainability_utils.integrated_gradients import generate_colored_text, get_top_n_attributions_dict
from utils.model_loader import load_from_pretrained_model_tokenizer, get_model_embeddings, get_prediction, binarize_predictions
from captum.attr import LayerIntegratedGradients
import torch
from captum.attr import visualization as viz
from ipymarkup import show_span_box_markup
from itertools import groupby
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#Define the class
class Ig(ExplainableModel):
    ## Define the constructor
    def __init__(self, model, tokenizer,text):
        """
        Constructor of the class
        :param model_path: path to the model
        :param tokenizer_id: tokenizer id
        """
        super().__init__(model, tokenizer, text)
        config = self.model.config
        model_type = config.model_type

        self.model_input = get_model_embeddings(self.model, model_type)
        self.model.to("cuda")
        self.labels_names=list(config.id2label.values())

        self.get_predictions()
        self.token_weights=self.compute_token_weights()
        print(self.token_weights)

    # def get_token_weights(self):
    #     return self.token_weights
    def get_predictions(self):
        # STEP 5: get predictions
        predictions = get_prediction(self.text, self.model, self.tokenizer, max_length=1512)
        # STEP 6: If there are predictions greater than 0.5 turn them to 1
        #Turn to 1 the 4 highest predictions
        predictions = binarize_predictions(predictions,top=True)
        self.predictions=predictions

    def model_output(self,inputs):
        return self.model(inputs)[0]

    def interpret_text(self,target_label, max_len=1512):
        """
        Given a text, a model, a tokenizer and a target label, return the attributions
        :param text: text to interpret
        :param model_input: model input
        :param model_output: model output function
        :param target_label: target label
        :param tokenizer: tokenizer
        :param max_len: max len
        :return: a list of pars (token, attribution)
        """
        # Instantiate the class to infer layer integrated gradients
        lig = LayerIntegratedGradients(self.model_output, self.model_input)

        # Call a function that extrats input ids, baseline_input_ids and all tokens
        input_ids, baseline_input_ids, all_tokens, self.token_ids2word = self.construct_input_and_baseline(max_len=1512)

        # Computes the integrated layers given the input_ids and baseline input_ids
        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=baseline_input_ids,
                                            return_convergence_delta=True,
                                            internal_batch_size=1,
                                            target=target_label
                                            )

        # The attributions must be normalized, we call the summarize attributions func
        attributions_sum = self.summarize_attributions(attributions)

        attributions_sum = attributions_sum[1:]
        num_attributions = len(all_tokens)
        attributions_sum = attributions_sum[:num_attributions]
        attributions_sum = attributions_sum.cpu().detach().numpy()
        attributions = list(zip(all_tokens, attributions_sum))
        return attributions

    def construct_input_and_baseline(self, max_len):
        """
        Construct the input and baseline tensors for the integrated gradients algorithm
        :param text: text to interpret
        :param max_len: max len
        :return: input_ids, baseline_ids and token list
        """
        max_length = max_len
        baseline_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.sep_token_id
        cls_token_id = self.tokenizer.cls_token_id

        # Encode text to get text_ids
        text_ids = self.tokenizer.encode(self.text, max_length=max_length, truncation=True, add_special_tokens=False)
        encoded = self.tokenizer(self.text, max_length=max_length, truncation=True, add_special_tokens=False, return_tensors="pt")

        token_ids2words = []
        indexes_list = list(range(len(text_ids)))
        words2token_ids_dict = dict(zip(indexes_list, encoded.word_ids()))

        grouped_keys = {k: [key for key, _ in v] for k, v in
                        groupby(words2token_ids_dict.items(), key=lambda item: item[1])}

        # Create the final list
        token_ids2words = list(grouped_keys.values())

        # Get the input ids and tokens list
        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        self.token_list = self.tokenizer.convert_ids_to_tokens(text_ids)

        # Generate baseline input_ids
        baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]

        # Return input_ids, baseline_ids and token list
        return torch.tensor([input_ids], device='cuda'), torch.tensor([baseline_input_ids],
                                                                      device='cuda'), self.token_list, token_ids2words

    def summarize_attributions(self,attributions):
        """
        Summarize the attributions
        :param attributions: attributions
        :return: summarized attributions
        """
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def compute_token_weights(self):
        """
        Get the attributions
        :return: attributions
        """
        predictions_indexes = self.predictions.tolist()[0]
        predictions_indexes = [index for (index, item) in enumerate(predictions_indexes) if item == 1.0]

        # STEP 8: For each prediction index that are 1 get the attributions
        attributions_dict = {}
        for prediction_index in predictions_indexes:
            tokens_attributions =self.interpret_text(prediction_index)
            tokens_attributions = [(token[0].replace("Ġ", ""),token[1]) for token in tokens_attributions]
            tokens_attributions = [(token[0].replace('Ċ', ""),token[1]) for token in tokens_attributions]

            attributions_dict[self.labels_names[prediction_index]] = tokens_attributions

        self.token_weights=attributions_dict

        return attributions_dict

    # def get_word_attributions(self,tokens_attributions):
    #     """
    #     Given the tokens attributions and the list of list with the tokens that correspond to each word, return a list with
    #     the attributions of each word
    #     :param tokens_attributions: list of pairs (token, attribution)
    #     :param token_ids2word: list of list with the tokens that correspond to each word
    #     :return: list with the attributions of each word
    #     """
    #     word_attributions = []
    #     for list in self.token_ids2word:
    #         attributions = 0.0
    #         for index in list:
    #             attributions += tokens_attributions[index][1]
    #         word_attributions.append(attributions)
    #     return word_attributions

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
    #         previous_index = -1
    #         for index in list:
    #             if previous_index != -1 and previous_index + 1 != index:
    #                 previous_index = previous_index + 1
    #                 while previous_index != index:
    #                     reconstructed_text += self.token_list[previous_index]
    #                     previous_index = previous_index + 1
    #             reconstructed_text += self.token_list[index]
    #             previous_index = index
    #
    #         reconstructed_text += " "
    #     reconstructed_text = reconstructed_text.replace("Ġ", "")
    #     reconstructed_text = reconstructed_text.replace('Ċ', "")
    #
    #     words_list = reconstructed_text.split(" ")
    #
    #     return reconstructed_text, words_list

    # def get_by_word_weights(self):
    #     """
    #     Given the tokens attributions and the list of list with the tokens that correspond to each word, return a dictionary
    #     with the attributions of each word
    #     :param tokens_attributions: list with the attributions of each token
    #     :param token_list: list with the tokens
    #     :param token_ids2word: list of list with the tokens that correspond to each word
    #     :return: list of tuples with the attributions of each word (word, attribution)
    #     """
    #     self.word_weights = {}
    #     for key in self.token_weights.keys():
    #         label_tokens_weights = self.token_weights[key]
    #         # Compute word weights
    #         label_word_weights = self.get_word_weights(label_tokens_weights)
    #
    #         # Get the word list
    #         reconstructed_text, words_list = self.get_reconstructed_text()
    #         label_word_weights = list(zip(words_list, label_word_weights))
    #
    #         # Add to the dictionary
    #         self.word_weights[key] = label_word_weights
    #
    #     # Average the weights of repeated words
    #     by_label_word_weights_dict = {}
    #     for key in self.word_weights.keys():
    #         label_word_weights = self.word_weights[key]
    #         word_weights_dict = {}
    #         for word, attribution in label_word_weights:
    #             if word in word_weights_dict.keys():
    #                 word_weights_dict[word].append(attribution)
    #             else:
    #                 word_weights_dict[word] = [attribution]
    #         for word in word_weights_dict.keys():
    #             word_weights_dict[word] = sum(word_weights_dict[word]) / len(word_weights_dict[word])
    #         by_label_word_weights_dict[key] = word_weights_dict
    #
    #     # Update the word weights list of lists
    #
    #     for key in self.word_weights.keys():
    #         label_word_weights = self.word_weights[key]
    #         word_weights_dict = by_label_word_weights_dict[key]
    #         for i in range(len(label_word_weights)):
    #             word = label_word_weights[i]
    #             attribution = word_weights_dict[word[0]]
    #             label_word_weights[i] = (word[0], attribution)
    #
    #     return self.word_weights, by_label_word_weights_dict