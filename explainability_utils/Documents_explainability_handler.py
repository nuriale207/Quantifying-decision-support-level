import json
import os

from explainability_utils.Ig import Ig
from explainability_utils.Lime import Lime
from explainability_utils.Random import Random
from explainability_utils.Shap import Shap
from explainability_utils.evaluation_utils import get_by_label_docs_explainability_degree
from explainability_utils.word_analysis_utils import get_top_word_weights, get_top_n_grams, get_top_html, \
    get_top_n_grams_weights

import numpy as np
class Documents_explainability_handler:

    def __init__(self,model,tokenizer,texts,method,by_file=False,language="english"):
        self.model=model
        self.tokenizer=tokenizer
        self.texts=texts
        self.method=method
        self.language=language
        #Compute the first step
        self.docs_top_n_grams_weights=None
        self.docs_top_word_weights=None
        if not by_file and not self.method=="lime":
            self.docs_tokens_lists,docs_words_weights=self.get_docs_by_word_weights()

    def get_docs_by_word_weights(self):
        """
        Get the word weights and token list of each document
        :return: dictionary with the word weights and token list of each document
        """
        docs_tokens_lists=[]
        docs_words_weights=[]
        for i in range(len(self.texts)):
            explainability_object=self.get_explainability_object(self.texts[i])
            # Step 1:Get token list and word attributions dictionary
            token_list, word_attributions_dict = explainability_object.get_by_word_weights()
            docs_tokens_lists.append(token_list)
            docs_words_weights.append(word_attributions_dict)
            if i%10==0:
                print("Document ",i," processed")
        self.docs_tokens_lists=docs_tokens_lists
        self.docs_words_weights=docs_words_weights

        return docs_tokens_lists,docs_words_weights

    def get_docs_top_word_weights(self,top_N_words=10,clean_stopwords=False,language="english"):
        """
        Get the top word weights of each document
        :param top_N_words: number of top words to return
        :param clean_stopwords: whether to clean stopwords or not
        :param language: language of the stopwords
        :return: dictionary with the top word weights of each document
        """
        if self.method=="lime":
            docs_top_word_weights=[]
            docs_tokens_lists = []
            docs_words_weights = []

            for i in range(len(self.texts)):
                explainability_object = self.get_explainability_object(self.texts[i])
                token_list, word_attributions_dict = explainability_object.get_by_word_weights()
                docs_tokens_lists.append(token_list)
                docs_top_word_weights.append(word_attributions_dict)
                docs_words_weights.append(word_attributions_dict)
            self.docs_top_word_weights=docs_top_word_weights
            self.docs_tokens_lists=docs_tokens_lists
            self.docs_words_weights=docs_words_weights

        else:
            docs_top_word_weights=[]
            for i in range(len(self.texts)):
                token_list=self.docs_tokens_lists[i]
                word_attributions_dict=self.docs_words_weights[i]
                top_word_weights=get_top_word_weights(word_attributions_dict, top_N_words, clean_stopwords=clean_stopwords,language=language)
                docs_top_word_weights.append(top_word_weights)
            self.docs_top_word_weights=docs_top_word_weights
            return docs_top_word_weights

    def get_docs_top_n_grams_weights(self,n=4):
        """
        Get the top n-grams weights of each document
        :param n: number of n-grams to return
        :return: dictionary with the top n-grams weights of each document
        """
        if self.docs_top_word_weights is None:
            self.get_docs_top_word_weights()

        docs_top_n_grams_weights=[]
        for i in range(len(self.texts)):
            token_list=self.docs_tokens_lists[i]
            word_attributions_dict=self.docs_words_weights[i]
            top_word_weights=self.docs_top_word_weights[i]
            top_n_grams_weights=get_top_n_grams_weights(token_list,word_attributions_dict, top_word_weights, n=n)
            docs_top_n_grams_weights.append(top_n_grams_weights)
        self.docs_top_n_grams_weights=docs_top_n_grams_weights
        return docs_top_n_grams_weights

    def get_explainability_object(self, text):
        """
        Get the explainability object according to the method
        :param text: text to explain
        :return: explainability object
        """
        if self.method=="shap":
            explainability_object = Shap(self.model,self.tokenizer,[text])
        elif self.method=="lime":
            explainability_object = Lime(self.model,self.tokenizer,text)
        elif self.method == "random":
            explainability_object = Random(self.model, self.tokenizer, text)
        else:
            explainability_object = Ig(self.model,self.tokenizer,text)

        return explainability_object

    def generate_docs_explainability_json(self, output_dir=None,save=True):
        """
        Generate a JSON file with the explainability results
        :param output_dir: output directory
        :return: JSON file
        """
        docs_explainability_json=[]
        for i in range(len(self.texts)):
            # 1-Get the token list
            token_list=self.docs_tokens_lists[i]
            # 2-Get the text from the token list
            text=" ".join(token_list)
            # 3-Get the word weights
            word_weights=self.docs_words_weights[i]
            # 6-Generate the JSON
            doc_explainability_json = {"text": text, "word_weights": word_weights}
            # 4-Get the top word weights in case they are computed
            if self.docs_top_word_weights is not None:
                top_word_weights=self.docs_top_word_weights[i]
                doc_explainability_json["top_word_weights"]=top_word_weights

            # 5-Get the top n-grams weights in case they are computed
            if self.docs_top_n_grams_weights is not None:
                top_n_grams_weights=self.docs_top_n_grams_weights[i].copy()
                # #Convert the n-grams to string
                # for key in top_n_grams_weights.keys():
                #
                #     top_n_grams_weights[key]={" ".join(item):value for item,value in top_n_grams_weights[key].items()}

                doc_explainability_json["top_n_grams_weights"]=top_n_grams_weights
            # 7-Append the JSON to the list
            docs_explainability_json.append(doc_explainability_json)

        # 8-Save the JSON
        if save and output_dir is not None:
            with open(output_dir, 'w') as outfile:
                json.dump(docs_explainability_json, outfile)

        return docs_explainability_json

    def load_from_json(self,explainability_json):
        """
        Load the explainability results from a JSON file
        :param explainability_json: JSON file
        :return: explainability object
        """
        docs_tokens_lists=[]
        docs_words_weights=[]
        docs_top_word_weights=[]
        docs_top_n_grams_weights=[]
        for doc_explainability_json in explainability_json:
            # 1-Get the token list
            token_list=doc_explainability_json["text"].split()
            docs_tokens_lists.append(token_list)
            # 2-Get the word weights
            word_weights=doc_explainability_json["word_weights"]
            docs_words_weights.append(word_weights)
            # 3-Get the top word weights in case they are computed
            if "top_word_weights" in doc_explainability_json.keys():
                top_word_weights=doc_explainability_json["top_word_weights"]
                docs_top_word_weights.append(top_word_weights)
            # 4-Get the top n-grams weights in case they are computed
            if "top_n_grams_weights" in doc_explainability_json.keys():
                top_n_grams_weights=doc_explainability_json["top_n_grams_weights"]
                docs_top_n_grams_weights.append(top_n_grams_weights)

        self.docs_tokens_lists=docs_tokens_lists
        self.docs_words_weights=docs_words_weights
        self.docs_top_word_weights=docs_top_word_weights
        self.docs_top_n_grams_weights=docs_top_n_grams_weights

        return docs_tokens_lists,docs_words_weights,docs_top_word_weights,docs_top_n_grams_weights

    def save_docs_html(self,output_dir,by_label=False):
        """
        Save an html file with the explainability results of each document
        :param output_dir: output directory
        :param by_label: whether to save the html by label or not
        :return: html file
        """

        for i in range(len(self.texts)):
            html=""
            #1-Get general top word weights html
            html+="<h3> Top word weights </h3> \n"
            top_word_weights_html=get_top_html(self.docs_tokens_lists[i],self.docs_top_word_weights[i],by_label=False)
            html+=top_word_weights_html +" \n"
            #1.2- If by label, get top word weights html by label
            if by_label:
                html += "<h3> By label top word weights </h3> \n"
                top_word_weights_html_by_label=get_top_html(self.docs_tokens_lists[i],self.docs_top_word_weights[i],by_label=True)
                html+=top_word_weights_html_by_label +" \n"

            #2-Get top n-grams weights html if they are computed
            if self.docs_top_n_grams_weights is not None:
                html+="<h3> Top n-grams weights </h3> \n"
                top_n_grams_weights_html=get_top_html(self.docs_tokens_lists[i],self.docs_top_n_grams_weights[i],by_label=False,n_grams=True)
                html+=top_n_grams_weights_html
                # 1.2- If by label, get top word n_grams html by label
                if by_label:
                    html += "<h3> By label top n-grams weights </h3> \n"
                    top_n_grams_weights_html_by_label = get_top_html(self.docs_tokens_lists[i],self.docs_top_n_grams_weights[i], by_label=True,n_grams=True)
                    html += top_n_grams_weights_html_by_label + " \n"

            #Save the html
            with open(os.path.join(output_dir,"doc_"+str(i)+".html"),"w") as f:
                f.write(html)


    def get_docs_explainability_degree(self, save=False, heatmap_dir=None):
        """
        Get the explainability degree of each document by label
        :return: dictionary with the explainability degree of each document by label
        """
        self.docs_by_label_explainability_degree=[]
        for i in range(len(self.docs_tokens_lists)):
            top_word_weights=self.docs_top_word_weights[i]
            if heatmap_dir is not None:
                #Generate a directory to save the heatmaps
                heatmap_dir_i=os.path.join(heatmap_dir,"doc_"+str(i))

            if save:
                if not os.path.exists(heatmap_dir_i):
                    os.mkdir(heatmap_dir_i)
                doc_explainability_degree=get_by_label_docs_explainability_degree(top_word_weights, save_maps=save,
                                                                              maps_path=heatmap_dir_i,language=self.language)
            else:
                doc_explainability_degree=get_by_label_docs_explainability_degree(top_word_weights, save_maps=save,
                                                                              maps_path=None,language=self.language)
            self.docs_by_label_explainability_degree.append(doc_explainability_degree)

        return self.docs_by_label_explainability_degree

    def generate_docs_explainability_degree_report(self, output_dir=None,save=True):
        """
        Generate a txt report file with the explainability degree of each document and the average explainability degree
        :param output_dir: output directory
        :return: txt file
        """
        docs_explainability_degree= ""
        docs_explainability_degree_list=[]
        for i in range(len(self.docs_tokens_lists)):
            # 1-Get the document explainability degree
            document_eval=self.docs_by_label_explainability_degree[i]
            document_degree=document_eval["overall"]
            # 2-Append the document explainability degree to the list
            docs_explainability_degree_list.append(document_degree)
            #Write the document explainability degree
            docs_explainability_degree+="Document "+str(i)+" explainability degree: "+str(document_degree)+"\n"
        #Get the average explainability degree
        average_degree=np.mean(docs_explainability_degree_list)
        #Write the average explainability degree
        docs_explainability_degree+="Average explainability degree: "+str(average_degree)+"\n"

        # 8-Save the report

        if save and output_dir is not None:
            with open(output_dir, 'w') as outfile:
                #write file as txt
                outfile.write(docs_explainability_degree)

        return docs_explainability_degree
