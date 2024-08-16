import sys
sys.path.append('..')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from explainability_utils.word_analysis_utils import get_definition
# from statistics import max
import tensorflow_hub as hub
import tensorflow as tf

import numpy as np
import seaborn as sns

#Load USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# module_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
model_USE = hub.load(module_url)
#
# module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
# load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
# embed = hub.load(module_url, options=load_options)



def get_understandability_degree(top_label_words,model="USE",save_maps=False,maps_path=None):
    """
    Compute understandability degree from top label words
    :param top_label_words: dictionary of top label words
    :return: understandability degree
    """
    understandability=[]
    for label in top_label_words:
        #Get label definition
        label_definition=get_definition(label)
        #Get understandability degree
        understandability_degree=get_understandability_degree_by_words(label_definition,top_label_words[label],model,
                                                                       return_matrix=True)
        understandability.append(np.max(understandability_degree))



    return max(understandability)


def get_docs_understandability_degree(top_label_words_dict,model="USE"):
    undertandability_degree=[]
    for dictionary in top_label_words_dict:
        only_words_dict = {key: [item[0] for item in value] for key, value in dictionary.items()}

        understandability=get_understandability_degree(only_words_dict,model="USE")
        undertandability_degree.append(understandability)
    return max(undertandability_degree)

def get_docs_understandability_degree_report(top_label_words_dict,model="USE",save_maps=False,maps_path=None):
    undertandability_degree = []
    for dictionary in top_label_words_dict:
        if n_grams:
            only_words_dict = {key: [" ".join(item[0]) for item in value] for key, value in dictionary.items()}
        else:
            only_words_dict = {key: [item[0] for item in value] for key, value in dictionary.items()}

        understandability = get_understandability_degree(only_words_dict, model="USE",save_maps=save_maps,maps_path=maps_path)
        undertandability_degree.append(understandability)

    overall_understandability = max(undertandability_degree)
    docs_understandability=undertandability_degree
    #Generate report
    report = "Overall understandability degree: " + str(overall_understandability) + "\n"
    for i in range(len(docs_understandability)):
        report += "Document " + str(i) + " understandability degree: " + str(docs_understandability[i]) + "\n"

    return report

def get_understandability_degree_by_words(def_words, inferred_words,model="USE",return_matrix=False):
    """
    Compute cosine similarity between two sets of words for understandability degree
    :param def_words: list of words
    :param inferred_words: list of words
    :return: cosine similarity
    """
    def_embeddings=model_USE(def_words)

    inferred_embeddings=model_USE(inferred_words)

    similarity=cosine_similarity(inferred_embeddings,def_embeddings)

    if not return_matrix:
        # Step 1: Find the maximum value in each row
        # max_values_by_row = np.max(similarity, axis=1)

        # Step 2: Compute the mean of the maximum values
        # similarity_max = np.mean(max_values_by_row)
        similarity_max=np.mean(similarity)
        return similarity_max
    else:
        return similarity



def get_by_label_docs_explainability_degree(top_word_weights,save_maps=False,maps_path=None,language="english"):
    """
    Compute explainability degree by label
    :param top_word_weights: dictionary of top word weights
    :param save_maps: whether to save by label heatmaps or not
    :param maps_path: path to save heatmaps
    :return: a dictionary of the explainability degree by document
    """

    explainability_degree={}
    for label in top_word_weights:
        def_words=get_definition(label,language=language)
        words=list(top_word_weights[label].keys())
        if not save_maps:
            explainability=get_understandability_degree_by_words(def_words,words,model="USE",return_matrix=True)
            if language == "english":
                # max_values_by_row = np.max(explainability, axis=1)
                # explainability_degree[label] = np.mean(max_values_by_row)
                explainability_degree[label] = np.mean(explainability)

            else:
                explainability_degree[label] = np.mean(explainability)
        else:
            explainability = get_understandability_degree_by_words(def_words, words, model="USE",return_matrix=True)
            if language=="english":
                # max_values_by_row = np.max(explainability, axis=1)
                # explainability_degree[label] = np.mean(max_values_by_row)
                explainability_degree[label] = np.mean(explainability)


            else:
                explainability_degree[label] = np.mean(explainability)
            # explainability_degree[label] = np.max(explainability)
            #Save maps
            generate_explainability_degree_heatmap(explainability,def_words,words,maps_path+"/"+label+".png")
    doc_explainability_degree=np.mean(list(explainability_degree.values()))
    explainability_degree={"overall":doc_explainability_degree,"by_label":explainability_degree}
    return explainability_degree


def generate_explainability_degree_heatmap(sim_matrix,def_words,words, maps_path):
    # Create a heatmap
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 10))
    sns.heatmap(sim_matrix, annot=True, cmap="YlGnBu", fmt=".2f", square=True, xticklabels=def_words,
                yticklabels=words)

    # Add axis labels
    plt.xlabel("Diagnostic term words")
    plt.ylabel("Definition words")

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # Save the heatmap
    # plt.show()
    plt.savefig(maps_path)
