from captum.attr import LayerIntegratedGradients
import torch
from captum.attr import visualization as viz
from ipymarkup import show_span_box_markup
from itertools import groupby
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# import ipymarkup
# from ipymarkup import show_span_box_markup

from utils.model_loader import get_prediction, binarize_predictions


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions


def construct_input_and_baseline(text, tokenizer, max_len):
    """
    Construct the input and baseline tensors for the integrated gradients algorithm
    :param text: text to interpret
    :param tokenizer: tokenizer
    :param max_len: max len
    :return: input_ids, baseline_ids and token list
    """
    max_length = max_len
    baseline_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id

    # Encode text to get text_ids
    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)
    encoded= tokenizer(text,max_length=max_length, truncation=True, add_special_tokens=False, return_tensors="pt")

    token_ids2words  = []
    indexes_list = list(range(len(text_ids)))
    words2token_ids_dict = dict(zip(indexes_list,encoded.word_ids()))

    grouped_keys = {k: [key for key, _ in v] for k, v in groupby(words2token_ids_dict.items(), key=lambda item: item[1])}

    # Create the final list
    # token_ids2words = [grouped_keys[key] for key in grouped_keys]
    token_ids2words=list(grouped_keys.values())
    # for word_id in encoded.word_ids():
    #     if word_id is not None:
    #         start, end = encoded.word_to_tokens(word_id)
    #         if start == end - 1:
    #             tokens = [start]
    #         else:
    #             tokens = [start, end - 1]
    #         if len(token_ids2words) == 0 or token_ids2words[-1] != tokens:
    #             token_ids2words.append(tokens)


    print(token_ids2words)
    # Get the input ids and tokens list
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    token_list = tokenizer.convert_ids_to_tokens(input_ids)
    token_list_2=tokenizer.convert_ids_to_tokens(text_ids)

    # reconstructed_text = ""
    # for ls in token_ids2words:
    #     previous_index = -1
    #     for index in ls:
    #         if previous_index!=-1 and previous_index+1!=index:
    #             previous_index=previous_index+1
    #             while previous_index!=index:
    #                 reconstructed_text += token_list_2[previous_index]
    #                 previous_index=previous_index+1
    #         reconstructed_text += token_list_2[index]
    #         previous_index=index
    #
    #     reconstructed_text += " "
    # reconstructed_text=reconstructed_text.replace("Ġ", "")
    # reconstructed_text=reconstructed_text.replace('Ċ', "")


    # Generate baseline input_ids
    baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]

    # Return input_ids, baseline_ids and token list
    return torch.tensor([input_ids], device='cuda'), torch.tensor([baseline_input_ids], device='cuda'), token_list_2, token_ids2words


def interpret_text(text, model_input, model_output, target_label, tokenizer, max_len):
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
    lig = LayerIntegratedGradients(model_output, model_input)

    # Call a function that extrats input ids, baseline_input_ids and all tokens
    input_ids, baseline_input_ids, all_tokens, token_ids2word = construct_input_and_baseline(text, tokenizer,max_len)

    # Computes the integrated layers given the input_ids and baseline input_ids
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=baseline_input_ids,
                                        return_convergence_delta=True,
                                        internal_batch_size=1,
                                        target=target_label
                                        )

    # The attributions must be normalized, we call the summarize attributions func
    attributions_sum = summarize_attributions(attributions)
    print(attributions_sum)
    attributions_sum=attributions_sum[1:]
    num_attributions=len(all_tokens)
    attributions_sum=attributions_sum[:num_attributions]

    # score_vis = viz.VisualizationDataRecord(
    #     word_attributions=attributions_sum,
    #     pred_prob=torch.max(model(input_ids)[0]),
    #     pred_class=torch.argmax(model(input_ids)[0]).numpy(),
    #     true_class=true_class,
    #     attr_class=text,
    #     attr_score=attributions_sum.sum(),
    #     raw_input_ids=all_tokens,
    #     convergence_score=delta)
    #
    # html = viz.visualize_text([score_vis])
    # with open("data_" + str(target_label) + ".html", "w") as file:
    #     file.write(html.data)
    #Cast the attributions to numpy
    attributions_sum = attributions_sum.cpu().detach().numpy()
    attributions=list(zip(all_tokens,attributions_sum))
    return attributions, all_tokens, token_ids2word


def generate_spans(labels_tokens_dict):
    """
    Given a list of list with labels and tokens index, and a text, return a list of tuples with the start and end of each span
    and the label
    :param labels_tokens_dict: list of list with labels and tokens index
    :param text: text
    :return: html with colored text
    """
    spans=[]
    for label in labels_tokens_dict:
        for token in labels_tokens_dict[label]:
            if not token[0] == -1:
                spans.append((token[0],token[-1],label))
    return spans


def generate_colored_text(labels_tokens_dict,text):
    """
    Given a list of list with labels and tokens index, and a text, return a list of tuples with the start and end of each span
    and the label
    :param labels_tokens_dict: list of list with labels and tokens index
    :param text: text
    :return: html with colored text
    """
    spans=generate_spans(labels_tokens_dict)
    html = show_span_box_markup(text, spans)

    return html

def generate_by_label_colored_text(labels_tokens_dict,text):
    """
    Given a list of list with labels and tokens index, and a text, return a list of tuples with the start and end of each span
    and the label
    :param labels_tokens_dict: list of list with labels and tokens index
    :param text: text
    :return: html with colored text
    """
    html_list=[]
    for label in labels_tokens_dict:
        label_dict={label:labels_tokens_dict[label]}
        spans = generate_spans(label_dict)
        html = show_span_box_markup(text, spans)
        html_list.append(html)
    return html_list
def get_word_attributions(tokens_attributions, token_ids2word):
    """
    Given the tokens attributions and the list of list with the tokens that correspond to each word, return a list with
    the attributions of each word
    :param tokens_attributions: list of pairs (token, attribution)
    :param token_ids2word: list of list with the tokens that correspond to each word
    :return: list with the attributions of each word
    """
    word_attributions = []
    for list in token_ids2word:
        attributions = 0.0
        for index in list:
            attributions += tokens_attributions[index][1]
        word_attributions.append(attributions)
    return word_attributions

def get_reconstructed_text(token_list, token_ids2word):
    """
    Given the list of tokens and the list of list with the tokens that correspond to each word, return the reconstructed
    text
    :param token_list: list with the tokens
    :param token_ids2word: list of list with the tokens that correspond to each word
    :return: reconstructed text
    """
    reconstructed_text = ""
    for list in token_ids2word:
        previous_index = -1
        for index in list:
            if previous_index != -1 and previous_index + 1 != index:
                previous_index = previous_index + 1
                while previous_index != index:
                    reconstructed_text += token_list[previous_index]
                    previous_index = previous_index + 1
            reconstructed_text += token_list[index]
            previous_index = index

        reconstructed_text += " "
    reconstructed_text = reconstructed_text.replace("Ġ", "")
    reconstructed_text = reconstructed_text.replace('Ċ', "")

    words_list = reconstructed_text.split(" ")

    return reconstructed_text, words_list

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
def find_all_word_occurrences(text, word):
    """
    Given a text and a word, return a list of tuples with the start and end of each occurrence of the word in the text
    :param text: text
    :param word: word
    :return: list of tuples with the start and end of each occurrence of the word in the text
    """
    occurrences = []
    if len(word)>0:
        for match in re.finditer(r'\b{}\b'.format(re.escape(word)), text, flags=re.IGNORECASE):
            occurrences.append((match.start(), match.end()))
    return occurrences

def get_by_word_attributions(tokens_attributions, token_list, token_ids2word):
    """
    Given the tokens attributions and the list of list with the tokens that correspond to each word, return a dictionary
    with the attributions of each word
    :param tokens_attributions: list with the attributions of each token
    :param token_list: list with the tokens
    :param token_ids2word: list of list with the tokens that correspond to each word
    :return: list of tuples with the attributions of each word (word, attribution)
    """

    # Get word attributions using token_ids2word
    attributions = get_word_attributions(tokens_attributions, token_ids2word)
    # Get the word list
    reconstructed_text, words_list = get_reconstructed_text(token_list, token_ids2word)
    # Get word attributions list
    word_attributions = list(zip(words_list, attributions))
    # Clean stop words from list of tuples
    stop_words = set(stopwords.words('english'))

    word_attributions = [word for word in word_attributions if word[0].lower() not in stop_words]
    # Clean empty words from list of tuples
    word_attributions= [word for word in word_attributions if word[0] != ""]
    # Clean words with length 1 from list of tuples
    word_attributions= [word for word in word_attributions if len(word[0]) > 1]

    return word_attributions
def get_top_n_attributions_dict(text, model, model_input, model_output, tokenizer, max_len, labels_names, n=10, by_word=False):
    """
    Given a text, a model, a tokenizer, the max length of the text, the labels names and the number of top attributions
    to return, return a dictionary with the top n attributions of each label
    :param text: text to interpret
    :param model: model to interpret
    :param model_input: model input
    :param model_output: model output function
    :param tokenizer: tokenizer of the model
    :param max_len: max length of the text
    :param labels_names: list with the labels names
    :param n: number of top attributions to return
    :param by_word: if True return the top n attributions by word, if False return the top n attributions by token
    :return: dictionary with the top n attributions of each label
    """
    # STEP 5: get predictions
    predictions = get_prediction(text, model, tokenizer, max_length=max_len)

    # STEP 6: If there are predictions greater than 0.5 turn them to 1
    predictions=binarize_predictions(predictions)

    # STEP 7: Get the index of the predictions that are 1
    predictions_indexes=predictions.tolist()[0]
    predictions_indexes=[index for (index, item) in enumerate(predictions_indexes) if item == 1.0]

    # STEP 8: For each prediction index that are 1 get the attributions
    attributions_dict = {}
    for prediction_index in predictions_indexes:
        tokens_attributions, token_list, token_ids2word = interpret_text(text, model_input, model_output, prediction_index, tokenizer,
                                                  max_len)
        if by_word:
            # Get word attributions list of tuples
            word_attributions = get_by_word_attributions(tokens_attributions, token_list, token_ids2word)
            #Get the top n words with the highest attribution
            if len(word_attributions)>n:
                #Get the words with the highest attribution
                #Sort the list of tuples by the second element of the tuple (the attribution)
                top_n_words=sorted(word_attributions, key=lambda x:x[1], reverse=True)[:n]
                #Get the words
                highligth_words = [word[0] for word in top_n_words]

            else:
                highligth_words = [word[0] for word in word_attributions]

        else:
            # Get token attributions
            attributions=tokens_attributions
            # reconstructed_text, _=get_reconstructed_text(token_list, token_ids2word)
            token_list = [(token[0].replace("Ġ", ""),token[1]) for token in attributions]
            token_list = [(token[0].replace('Ċ', ""),token[1]) for token in token_list]

            #Get the top 10 words with the highest attribution
            #if there are more than 10 words
            if len(attributions)>10:
                #Get the words with the highest attribution from the lust of tuples
                top_n_words=sorted(token_list, key=lambda x:x[1], reverse=True)[:n]
                # Get the words
                highligth_words = [word[0] for word in top_n_words]
            else:
                highligth_words = [word[0] for word in attributions]
            # Get the tokens index in the text

            # highligth_words = [words_list[i] for i in top_n_words]

        words_index = []
        for word in highligth_words:
            # Find all the start and end index of the token
            find_all_word_occurrences(text, word)
            words_index.extend(find_all_word_occurrences(text, word))

        print(words_index)
        # if there are any (0,0) tuples remove them
        words_index = [x for x in words_index if x != (0, 0)]
        attributions_dict[labels_names[prediction_index]] = words_index

    return attributions_dict

