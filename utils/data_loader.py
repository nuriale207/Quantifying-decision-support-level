import pandas as pd

def load_data(csv_path,sep,labels_default_start_index, text_column_id="text"):
    """
    Load data from csv file

    Args:
        csv_path (str): path to csv file
        sep (str): separator used in csv file
        labels_default_start_index (int): index of first label column
        text_column_id (str): name of the text column in the csv file. By default "text" will be used

    Returns:
        texts (list): list of texts
        labels (list): list of labels
    """
    df=pd.read_csv(csv_path,sep=sep)

    df.columns = [x.lower() for x in df.columns]

    df_labels_columns= df.columns[labels_default_start_index:]

    #Return texts and labels as lists
    texts=df[text_column_id].tolist()
    labels=df[df_labels_columns].values.tolist()

    return texts,labels
