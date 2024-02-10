import copy
import os
import sys
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from multiligtopic.openai_api_key import OPENAI_API_KEY
from models.cnn.model import CNN
from datasets.scar import SCAR
import pandas as pd
from bertopic import BERTopic
import openai
from bertopic.representation import KeyBERTInspired, OpenAI
from umap import UMAP
from hdbscan import HDBSCAN


sys.path.insert(0, os.path.abspath('../'))


class BERTopicDeploy:

    def __init__(self, config):
        self.config = config
        self.tokenizer = get_tokenizer('basic_english')
        self.text_to_viz = []
        self.device = config.device
        self.target = config.target
        self.filt_sents_f = None
        self.topic_model = None
        self.top_topics = None

        # Read the file and store each line (sentence) in a list
        with open(config.load_file, 'r') as file:
            sentences = file.readlines()
        # Remove newline characters and any leading/trailing whitespace from each sentence
        self.sents = [sentence.strip() for sentence in sentences]
        self.filt_sents_f = config.load_file  # this is the filename for the filtered sentenc

    def fit_topic_model(self, docs):

        # Prepare UMAP
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        # Prepare HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)

        # Prepare Vectorizer
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

        # Prepare representations
        # KeyBERT
        keybert_model = KeyBERTInspired()
        # MMR
        # mmr_model = MaximalMarginalRelevance(diversity=0.3) # This is used by default for BERTopic
        # GPT-3.5
        openai.api_key = OPENAI_API_KEY  # Loaded from a file that I wont upload to git

        prompt = """
        I have a topic that contains the following documents:
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make 
        sure it is in the following format: topic: <topic label> """
        openai_model = OpenAI(model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

        if self.config.openai_flag == "send_to_openai":

            user_response = self.get_user_input()

            if user_response == "y":
                print("User has confirmed we can send data to Openai for summarization")
                representation_model = {
                    "KeyBERT": keybert_model,
                    "OpenAI": openai_model}
            else:
                print("User cancelled sending data to Openai for summarization")
                representation_model = {
                    "OpenAI": openai_model,
                }
        else:
            print("Openai flag not provided, will not use this for topic summarization")
            representation_model = {
                "OpenAI": openai_model,
            }
        print('Prepared BERTopic Components')

        # topics, probs = topic_model.fit_transform(docs)
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=10,
            nr_topics=21,  # The first one is the outlier topic, so need 21
            verbose=True
        )
        print('About to fit')
        topic_model.fit(docs)

        print('Fitted the topic model!')

        self.topic_model = topic_model

        # Write out the topic_model to a new file
        topic_f = self.filt_sents_f
        # Extract the directory, filename, and extension
        directory, full_filename = os.path.split(topic_f)
        filename_without_extension, extension = os.path.splitext(full_filename)
        # Replace "impt_sents_" with "topic_df_"
        new_filename_without_prefix = filename_without_extension.replace("impt_sents_", "raw_topic_df_")
        # Create the new filename with the .csv extension
        topic_df_filename = os.path.join(directory, new_filename_without_prefix + ".csv")

        topic_model.get_topic_info().to_csv(topic_df_filename, index=False)

    def extract_top_topics(self, n_top_topics):

        def extract_first_from_tuple_list(tuple_list):
            return [item[0] for item in tuple_list]

        topic_df = self.topic_model.get_topic_info().head(n_top_topics + 1)
        topic_df = topic_df.iloc[1:]
        topic_df = topic_df.set_index('Topic')
        topic_df['Words'] = topic_df.index.map(self.topic_model.get_topic)
        topic_df['Words'] = topic_df['Words'].apply(extract_first_from_tuple_list)

        self.top_topics = topic_df
        print(topic_df)

        # Write out the DataFrame to a new file
        topic_f = self.filt_sents_f
        # Extract the directory, filename, and extension
        directory, full_filename = os.path.split(topic_f)
        filename_without_extension, extension = os.path.splitext(full_filename)
        # Replace "impt_sents_" with "topic_df_"
        new_filename_without_prefix = filename_without_extension.replace("impt_sents_", "cleaned_topic_df_")
        # Create the new filename with the .csv extension
        topic_df_filename = os.path.join(directory, new_filename_without_prefix + ".csv")

        topic_df.to_csv(topic_df_filename)

        print(f'Printed topic_df to: {topic_df_filename}')

    @staticmethod
    def get_user_input():
        while True:
            user_input = input("About to send some data to Openai for topic summarization. \n"
                               "Type 'y' to proceed or 'n' to cancel: ").lower()
            if user_input in ['y', 'n']:
                return user_input
            else:
                print("Invalid input. Please enter 'y' or 'n'.")