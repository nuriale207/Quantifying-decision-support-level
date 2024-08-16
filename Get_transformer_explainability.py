import sys

from explainability_utils.Documents_explainability_handler import Documents_explainability_handler

sys.path.append("..")
import argparse

from utils import data_loader, model_loader
import os
import time
import json

#def main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for getting explainability of transformer models ",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Arguments
    #Data and model loading arguments
    parser.add_argument('--test_data_dir', type=str, default="test_data",
                             help="Directory containing the test dataset")
    parser.add_argument("--labels_default_start_index", type=int, default=None,
                        help="Default start index for the labels. If None, the labels will be indexed from 0 to n_labels-1")
    parser.add_argument("--text_column_mame", type=str, default="text",
                        help="Index of the text column in the csv file. If None, the first column will be used")
    parser.add_argument("--csv_sep", type=str, default="\t")

    parser.add_argument('--model_id', type=str, default="allenai/longformer-base-4096",
                             help="Pretrained model name")
    parser.add_argument('--model_type', type=str, default="longformer",
                             help="Pretrained model type")
    parser.add_argument('--model_path', type=str, default="model_id",
                                help="Directory containing the model to load if any model is provided")
    # add an argument called from_file that if it is present, the model will be loaded from the model_path
    parser.add_argument('--from_file', action='store_true',
                        help="Whether to load the model from the model_path. If not, the model will be loaded from the model_id")
    # add an argument called json_path that if from_file is present, the model will be loaded from the json_path
    parser.add_argument('--json_path', type=str, default="model_id",
                        help="Directory containing the model to load if any model is provided")

    #Explainability options arguments
    parser.add_argument("--language", type=str, default="english",
                        help="Language of the texts. Options: english or spanish. By default, english will be used")

    parser.add_argument("--explainability_method", type=str, default="all",
                        help="Name of the explainability method to use. Options: lime, shap, ig, all. "
                             "By default, all methods will be used")

    parser.add_argument("--n_highlighted_words", type=int, default=5,
                        help="Number of top words to highlight in the explainability results. By default, "
                             "5 words will be highlight")

    parser.add_argument("--no-stopwords", type=bool, default=False,
                        help="Whether not to remove stopwords in the explainability results. By default, False")

    parser.add_argument("--n-grams", type=bool, default=False,
                        help="Whether to use n-grams in the explainability results. By default, False")

    parser.add_argument("--n-grams_size", type=int, default=3,
                        help="Size of the n-grams to use in the explainability results. "
                             "By default, 3-grams will be used")

    parser.add_argument("--n-top-labels", type=int, default=4,
                        help="Number of top labels to return in the explainability results. "
                             "By default, 4 labels will be returned")

    #Output options arguments
    parser.add_argument('--output_dir', type=str, default="output",
                             help="Directory to save the generated files: predictions, explainability files, "
                                  "metrics, etc.")
    parser.add_argument("--experiment_name", type=str, default="longformer",
                        help="Name of the experiment")

    parser.add_argument('--no_return_html', type=bool, default=False,
                             help="Whether to not return an html file per instance with the explainability results. "
                                  "By default, False")

    parser.add_argument('--no_return_metrics', type=bool, default=False,
                                help="Whether to not return a metrics file with the explainability results. "
                                     "By default, False")

    #If html file is returned
    parser.add_argument('--html_graph_type', type=str, default="both",
                                help="Type of graph to return in the html file. Options: both, general, separated by "
                                     "label. By default, both will be returned")

    #Instantiate DataHolder
    args = parser.parse_args()

    #Start a timer
    starttime = time.time()

    #Get the data texts and labels in a list
    texts, labels = data_loader.load_data(args.test_data_dir, args.csv_sep, args.labels_default_start_index, args.text_column_mame)
    # texts=texts[:10]
    # texts=["This is a test sentence","This is another test sentence"]
    #Load the model and tokenizer
    if args.model_path=="model_id":
        args.model_path=args.model_id

    model, tokenizer = model_loader.load_from_pretrained_model_tokenizer(args.model_id, args.model_path)

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir_explainability = os.path.join(output_dir, "explainability")
    if not os.path.exists(output_dir_explainability):
        os.makedirs(output_dir_explainability)

    #For each technique, get the explainability results
    explainability_methods=args.explainability_method.split(",")
    explainability_methods=[x.strip() for x in explainability_methods]
    if explainability_methods[0]=="all":
        # explainability_methods=["ig","shap","lime"]
        explainability_methods=["random","ig","shap"]
    #generate an output directory for each explainability method
    dirs_dict={}
    for explainability_method in explainability_methods:
        dir_name=os.path.join(output_dir_explainability,explainability_method)
        dirs_dict[explainability_method]=dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    for explainability_method in explainability_methods:
        #If a file is not loaded
        if not args.from_file:
            #Print the explainability method
            print("Explainability method: ",explainability_method)

            #Get the explainability object
            explainabilty_handler=Documents_explainability_handler(model,tokenizer,texts,method=explainability_method)

            #Get the top words of each document
            docs_top_word_weights=explainabilty_handler.get_docs_top_word_weights(top_N_words=args.n_highlighted_words,
                                                                                  clean_stopwords=args.no_stopwords,
                                                                                  language=args.language)
            #If n-grams are used, get the top n-grams of each document
            if args.n_grams:
                docs_top_n_grams_weights=explainabilty_handler.get_docs_top_n_grams_weights(n=args.n_grams_size)

            #Generate and save a JSON file with the explainability results
            explainabilty_handler.generate_docs_explainability_json(output_dir=dirs_dict[explainability_method]+"/docs_explainability.json")

            #If html file is returned
            if not args.no_return_html:
                #Define the output directory
                html_output_dir=os.path.join(dirs_dict[explainability_method],"html")
                #If the directory does not exist, create it
                if not os.path.exists(html_output_dir):
                    print("Creating directory: ",html_output_dir)
                    os.makedirs(html_output_dir)
                #Save an html file with the explainability results of each document
                explainabilty_handler.save_docs_html(html_output_dir,by_label=True)
        else:
            #Load json from args.json_path
            # explainability_json=json.load(args.json_path)
            with open(args.json_path, 'r') as j:
                explainability_json = json.loads(j.read())
            #Load the explainability object from a file
            explainabilty_handler=Documents_explainability_handler(model,tokenizer,texts,method=explainability_method,by_file=True)
            explainabilty_handler.load_from_json(explainability_json)
        #If metrics file is returned
        if not args.no_return_metrics:
            #Define the output directory
            metrics_output_dir=os.path.join(dirs_dict[explainability_method],"metrics")
            #If the directory does not exist, create it
            if not os.path.exists(metrics_output_dir):
                print("Creating directory: ",metrics_output_dir)
                os.makedirs(metrics_output_dir)



            #Calculate the metrics of the explainability results of each document
            explainabilty_handler.get_docs_explainability_degree(save=True, heatmap_dir=metrics_output_dir)

            #Save the report and json with the explainability degree
            explainabilty_handler.generate_docs_explainability_degree_report(output_dir=metrics_output_dir+"/docs_explainability_degree_report.json",save=True)



        if explainability_method == "ig":
            model.to("cpu")

    #End the timer
    endtime = time.time()-starttime
    print("Time elapsed: ",endtime)