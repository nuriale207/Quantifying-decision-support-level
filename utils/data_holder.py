from transformers import AutoTokenizer

from utils.data_handler import DataHandler


class DataHolder:
    def __init__(self, args,hierarchical=False):
        self.args = args
        self.hierarchical=hierarchical
        #load arguments in self.args


        self.data_sep=self.args.csv_sep
        self.model_id = self.args.model_id
        self.model_type = self.args.model_type
        self.model_out = self.args.model_out
        self.output_dir = self.args.output_dir
        self.output_attentions = self.args.output_attentions
        self.no_cache = self.args.no_cache
        self.max_len= self.args.max_len
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.lr= self.args.lr
        self.warmup_proportion = self.args.warmup_proportion
        self.adam_epsilon = self.args.adam_epsilon
        self.weight_decay = self.args.weight_decay
        self.max_grad_norm = self.args.max_grad_norm
        self.metric_for_best_model = self.args.metric_for_best_model
        self.seed = self.args.seed

        if hierarchical:
            self.training_data_dir_full = self.args.training_data_dir_full
            self.test_data_dir_full = self.args.test_data_dir_full
            self.eval_data_dir_full = self.args.eval_data_dir_full
            self.labels_ids_full = self.args.labels_ids_full
            self.labels_default_start_index_full = self.args.labels_default_start_index_full

            self.training_data_dir_main = self.args.training_data_dir_main
            self.test_data_dir_main = self.args.test_data_dir_main
            self.eval_data_dir_main = self.args.eval_data_dir_main
            self.labels_ids_main = self.args.labels_ids_main
            self.labels_default_start_index_main = self.args.labels_default_start_index_main

            self.training_data_dir_chapter = self.args.training_data_dir_chapter
            self.test_data_dir_chapter = self.args.test_data_dir_chapter
            self.eval_data_dir_chapter = self.args.eval_data_dir_chapter
            self.labels_ids_chapter = self.args.labels_ids_chapter
            self.labels_default_start_index_chapter = self.args.labels_default_start_index_chapter


        else:
            self.training_data_dir = self.args.training_data_dir
            self.test_data_dir = self.args.test_data_dir
            self.eval_data_dir = self.args.eval_data_dir
            self.labels_ids = self.args.labels_ids
            self.labels_default_start_index = self.args.labels_default_start_index

    def load_data(self):
        if self.hierarchical:
            return self.load_data_hierarchical()
        else:
            #instantiate data handler
            tokenizer=AutoTokenizer.from_pretrained(self.model_id)

            print("Loading training data...")
            training_data_handler=DataHandler(self.training_data_dir,self.batch_size,
                                                   self.labels_ids,tokenizer,self.max_len, self.labels_default_start_index,
                                              self.data_sep)
            print("Loading test data...")
            test_data_handler=DataHandler(self.test_data_dir,self.batch_size,
                                                      self.labels_ids,tokenizer,self.max_len, self.labels_default_start_index,
                                          self.data_sep)

            print("Loading training data...")

            dev_data_handler=DataHandler(self.eval_data_dir,self.batch_size,
                                                        self.labels_ids,tokenizer,self.max_len, self.labels_default_start_index,
                                          self.data_sep)

            #load data
            self.training_data=training_data_handler.get_tokenized_data()
            self.test_data=test_data_handler.get_tokenized_data()
            self.dev_data=dev_data_handler.get_tokenized_data()
            label2id, id2label = training_data_handler.get_labels_dict()
            return self.training_data,self.dev_data,self.test_data,label2id, id2label

    def generate_model(self):
        self.model=AutoModelForSequenceClassification.from_pretrained(self.model_id)

    def load_data_hierarchical(self):

        #instantiate data handler
        tokenizer=AutoTokenizer.from_pretrained(self.model_id)

        print("Loading training data...")
        training_data_handler_main=DataHandler(self.training_data_dir_main,self.batch_size,
                                                  self.labels_ids_main,tokenizer,self.max_len, self.labels_default_start_index_main,
                                                self.data_sep)
        training_data_handler_chapter=DataHandler(self.training_data_dir_chapter,self.batch_size,
                                                    self.labels_ids_chapter,tokenizer,self.max_len, self.labels_default_start_index_chapter,
                                                    self.data_sep)
        training_data_handler_full=DataHandler(self.training_data_dir_full,self.batch_size,
                                                    self.labels_ids_full,tokenizer,self.max_len, self.labels_default_start_index_full,
                                                    self.data_sep)

        print("Loading test data...")
        test_data_handler_main=DataHandler(self.test_data_dir_main,self.batch_size,
                                                    self.labels_ids_main,tokenizer,self.max_len, self.labels_default_start_index_main,
                                                    self.data_sep)
        test_data_handler_chapter=DataHandler(self.test_data_dir_chapter,self.batch_size,
                                                    self.labels_ids_chapter,tokenizer,self.max_len, self.labels_default_start_index_chapter,
                                                    self.data_sep)
        test_data_handler_full=DataHandler(self.test_data_dir_full,self.batch_size,
                                                    self.labels_ids_full,tokenizer,self.max_len, self.labels_default_start_index_full,
                                                    self.data_sep)
        print("Loading eval data...")
        eval_data_handler_main=DataHandler(self.eval_data_dir_main,self.batch_size,
                                                    self.labels_ids_main,tokenizer,self.max_len, self.labels_default_start_index_main,
                                                    self.data_sep)
        eval_data_handler_chapter=DataHandler(self.eval_data_dir_chapter,self.batch_size,
                                                    self.labels_ids_chapter,tokenizer,self.max_len, self.labels_default_start_index_chapter,
                                                    self.data_sep)
        eval_data_handler_full=DataHandler(self.eval_data_dir_full,self.batch_size,
                                                    self.labels_ids_full,tokenizer,self.max_len, self.labels_default_start_index_full,
                                                    self.data_sep)

        #load data
        self.training_data_main=training_data_handler_main.get_tokenized_data()
        self.training_data_chapter=training_data_handler_chapter.get_tokenized_data()
        self.training_data_full=training_data_handler_full.get_tokenized_data()

        self.test_data_main=test_data_handler_main.get_tokenized_data()
        self.test_data_chapter=test_data_handler_chapter.get_tokenized_data()
        self.test_data_full=test_data_handler_full.get_tokenized_data()

        self.eval_data_main=eval_data_handler_main.get_tokenized_data()
        self.eval_data_chapter=eval_data_handler_chapter.get_tokenized_data()
        self.eval_data_full=eval_data_handler_full.get_tokenized_data()

        label2id_main, id2label_main = training_data_handler_main.get_labels_dict()
        label2id_chapter, id2label_chapter = training_data_handler_chapter.get_labels_dict()
        label2id_full, id2label_full = training_data_handler_full.get_labels_dict()

        return self.training_data_main,self.training_data_chapter,self.training_data_full,self.eval_data_main,self.eval_data_chapter,self.eval_data_full,self.test_data_main,self.test_data_chapter,self.test_data_full,label2id_main, id2label_main,label2id_chapter, id2label_chapter,label2id_full, id2label_full

