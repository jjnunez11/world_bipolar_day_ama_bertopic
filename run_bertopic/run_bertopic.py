import csv
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from run_bertopic.openai_api_key import OPENAI_API_KEY
from bertopic import BERTopic
import openai
from bertopic.representation import KeyBERTInspired, OpenAI

from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

from collections import defaultdict

from sklearn.datasets import fetch_20newsgroups

sys.path.insert(0, os.path.abspath('../'))


class RunBERTopic:

    def __init__(self, args, f, results_dir):
        self.f = f
        self.args = args
        self.text_to_viz = []
        self.device = args.device
        self.filt_sents_f = None
        self.topic_model = None
        self.top_topics = None
        self.results_dir = results_dir
        self.n_top_docs = args.n_top_docs

        # Read the file and store each line (sentence) in a list
        with open(self.f, 'r', encoding='utf-8') as file:
            self.documents = file.readlines()
        # Remove newline characters and any leading/trailing whitespace from each sentence
        # self.sents = [sentence.strip() for sentence in sentences]

        # For testing, uses a publicaly available dataset
        # self.documents = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))["data"][0:2000]

        print(f'We have loaded {len(self.documents)} docuemnts for analysis')

    def fit_topic_model(self):

        # Prepare UMAP
        umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=42)  # min_dist=0.1

        print("UMAP prepared!")

        # Prepare HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom',
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
        #openai_model = OpenAI(model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)
        openai_model = OpenAI(model="gpt-4o", exponential_backoff=True, chat=True, prompt=prompt)

        if self.args.openai_flag == "send_to_openai":

            user_response = self.get_user_input()

            if user_response == "y":
                print("User has confirmed we can send data to Openai for summarization")
                representation_model = {
                    "KeyBERT": keybert_model,
                    "OpenAI": openai_model}
            else:
                print("User cancelled sending data to Openai for summarization")
                representation_model = {
                    "KeyBERT": keybert_model,
                }
        else:
            print("Openai flag not provided, will not use this for topic summarization")
            representation_model = {
                "KeyBERT": keybert_model,
            }
        print('Prepared BERTopic Components')

        # Fit the topic model
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

        topic_model.fit(self.documents)

        print('Fitted the topic model!')

        self.topic_model = topic_model

        # Create the new filename with the .csv extension
        args = self.args
        topic_df_filename = os.path.join(self.results_dir, f"topics_{args.doc_types}_{args.years}.csv")

        print(f'Here is the filename to save to: {topic_df_filename}')

        topic_model.get_topic_info().to_csv(topic_df_filename, index=False)

    @staticmethod
    def get_user_input():
        while True:
            user_input = input("About to send some data to Openai for topic summarization. \n"
                               "Type 'y' to proceed or 'n' to cancel: ").lower()
            if user_input in ['y', 'n']:
                return user_input
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    import os
    import pandas as pd

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

        # Extract the top n representative documents

        # Prepare your documents to be used in a dataframe
        doc_df = pd.DataFrame({"Document": self.documents,
                               "ID": range(len(self.documents)),
                               "Topic": self.topic_model.topics_})

        repr_docs_mappings, repr_docs, repr_docs_indices, repr_docs_ids = self.topic_model._extract_representative_docs(
            c_tf_idf=self.topic_model.c_tf_idf_,
            documents=doc_df,
            topics=self.topic_model.topic_representations_,
            nr_repr_docs=self.n_top_docs
        )

        # Define file path for the combined CSV
        topics_docs_filename = os.path.join(self.results_dir,
                                            f"topics_and_documents_{self.args.doc_types}_{self.args.years}.csv")

        # Create a combined CSV for topics and documents, filtering out Topic -1
        with open(topics_docs_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Topic', 'Document'])  # Write the header
            for mapping, indices in zip(repr_docs_mappings, repr_docs_indices):
                if mapping != -1:  # Filter out Topic -1
                    for index in indices:
                        writer.writerow([mapping, repr_docs[index]])

        print(f'Printed topics and documents to: {topics_docs_filename}')
