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
        openai_model = OpenAI(model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

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

        # Write out the DataFrame to a new file
        args = self.args
        topic_df_filename = os.path.join(self.results_dir, f"cleaned_topics_{args.doc_types}_{args.years}.csv")
        topic_df.to_csv(topic_df_filename)

        print(f'Printed topic_df to: {topic_df_filename}')

        # Get and print document info DataFrame
        doc_info_df = self.topic_model.get_document_info(self.documents)
        print(doc_info_df)

        # Write out the document info DataFrame to a new file
        doc_info_filename = os.path.join(self.results_dir, f"document_info_{args.doc_types}_{args.years}.csv")
        doc_info_df.to_csv(doc_info_filename)

        print(f'Printed document_info_df to: {doc_info_filename}')

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

        # Check lengths of all arrays
        len_mappings = len(repr_docs_mappings)
        len_docs = len(repr_docs)
        len_indices = len(repr_docs_indices)
        len_ids = len(repr_docs_ids)

        print(f"Lengths - Mappings: {len_mappings}, Docs: {len_docs}, Indices: {len_indices}, IDs: {len_ids}")

        """
        if len_mappings == len_docs == len_indices == len_ids:
            # Combine the outputs into a single DataFrame
            repr_docs_df = pd.DataFrame({
                'Document': repr_docs,
                'Mapping': repr_docs_mappings,
                'Index': repr_docs_indices,
                'ID': repr_docs_ids
            })

            # Write out the combined DataFrame to a new file
            top_n_docs_filename = os.path.join(self.results_dir,
                                               f"top_{self.n_top_docs}_docs_{args.doc_types}_{args.years}.csv")
            repr_docs_df.to_csv(top_n_docs_filename, index=False)

            print(f'Printed top n docs to: {top_n_docs_filename}')
        else:
            print(
                "Error: The lengths of the arrays are not equal. Please check the output of _extract_representative_docs.")
        """

        # Write out the metadata to a new file without using pandas
        meta_filename = os.path.join(self.results_dir, f"top_{self.n_top_docs}_meta_{args.doc_types}_{args.years}.csv")
        with open(meta_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Mapping', 'Index', 'ID'])  # Write the header
            for mapping, index, doc_id in zip(repr_docs_mappings, repr_docs_indices, repr_docs_ids):
                writer.writerow([mapping, index, doc_id])

        print(f'Printed top n docs metadata to: {meta_filename}')

        # Create a CSV file for the documents
        docs_filename = os.path.join(self.results_dir, f"top_{self.n_top_docs}_docs_{args.doc_types}_{args.years}.csv")
        with open(docs_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Document'])  # Write the header
            for doc in repr_docs:
                writer.writerow([doc])

        print(f'Printed top n docs to: {docs_filename}')
