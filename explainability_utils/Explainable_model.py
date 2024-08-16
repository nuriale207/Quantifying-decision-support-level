
#Class explainable model parent class of all explainable models: Ig, Lime, Shap

class ExplainableModel:
    def __init__(self, model, tokenizer, text):
        self.model = model
        self.tokenizer = tokenizer
        self.text = text

        self.token_weights = {}
        self.token_list = []
        self.token_ids2word = []

    def get_token_weights(self):
        return self.token_weights

    def compute_token_weights(self):
        pass

    def get_word_weights(self, tokens_weights):
        """
        Given the tokens weights and the list of list with the tokens that correspond to each word, return a list with
        the weights of each word
        :param tokens_weights: list of pairs (token, attribution)
        :param token_ids2word: list of list with the tokens that correspond to each word
        :return: list with the weights of each word
        """
        word_weights = []
        for list in self.token_ids2word:
            weights = 0.0
            for index in list:
                weights += tokens_weights[index][1]
            word_weights.append(weights)
        return word_weights

    def get_reconstructed_text(self):
        """
        Given the list of tokens and the list of list with the tokens that correspond to each word, return the reconstructed
        text
        :param token_list: list with the tokens
        :param token_ids2word: list of list with the tokens that correspond to each word
        :return: reconstructed text
        """
        reconstructed_text = ""
        words_list = []
        for list in self.token_ids2word:
            previous_index = -1
            reconstructed_word = ""
            for index in list:
                if previous_index != -1 and previous_index + 1 != index:
                    previous_index = previous_index + 1
                    while previous_index != index:
                        reconstructed_text += self.token_list[previous_index]
                        reconstructed_word += self.token_list[previous_index]
                        previous_index = previous_index + 1
                else:
                    reconstructed_word+=self.token_list[index]
                reconstructed_text += self.token_list[index]
                previous_index = index
            reconstructed_word = reconstructed_word.replace("Ġ", "")
            reconstructed_word = reconstructed_word.replace('Ċ', "")
            words_list.append(reconstructed_word)
            reconstructed_text += " "
        reconstructed_text = reconstructed_text.replace("Ġ", "")
        reconstructed_text = reconstructed_text.replace('Ċ', "")

        # words_list = reconstructed_text.split(" ")

        return reconstructed_text, words_list

    def get_by_word_weights(self):
        """
        Given the tokens attributions and the list of list with the tokens that correspond to each word, return a dictionary
        with the attributions of each word
        :param tokens_attributions: list with the attributions of each token
        :param token_list: list with the tokens
        :param token_ids2word: list of list with the tokens that correspond to each word
        :return: list of tuples with the attributions of each word (word, attribution)
        """
        self.word_weights = {}
        for key in self.token_weights.keys():
            label_tokens_weights = self.token_weights[key]
            # Compute word weights
            label_word_weights = self.get_word_weights(label_tokens_weights)

            # Get the word list
            reconstructed_text, words_list = self.get_reconstructed_text()

            label_word_weights = list(zip(words_list, label_word_weights))

            # Add to the dictionary
            self.word_weights[key] = label_word_weights

        # Average the weights of repeated words
        by_label_word_weights_dict = {}
        for key in self.word_weights.keys():
            label_word_weights = self.word_weights[key]
            word_weights_dict = {}
            for word, attribution in label_word_weights:
                if word in word_weights_dict.keys():
                    word_weights_dict[word].append(attribution)
                else:
                    word_weights_dict[word] = [attribution]
            for word in word_weights_dict.keys():
                word_weights_dict[word] = sum(word_weights_dict[word]) / len(word_weights_dict[word])
            by_label_word_weights_dict[key] = word_weights_dict

        # Update the word weights list of lists

        for key in self.word_weights.keys():
            label_word_weights = self.word_weights[key]
            word_weights_dict = by_label_word_weights_dict[key]
            for i in range(len(label_word_weights)):
                word = label_word_weights[i]
                attribution = word_weights_dict[word[0]]
                label_word_weights[i] = (word[0], attribution)
        #Get token list
        self.token_list = [word[0] for word in self.word_weights[key]]

        return self.token_list, by_label_word_weights_dict