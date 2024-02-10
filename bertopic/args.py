import os
from argparse import ArgumentParser


def get_args():

    # Please note: we plan to add arguments to allow more customization of the BERTopic steps

    parser = ArgumentParser(description="Deployment of BERTopic to find topics of Reddit AMA ")

    parser.add_argument('--openai-flag', default='', type=str, choices=['', "send_to_openai"],
                        help="String should be send_to_openai if this is desired for topic summarization")

    parser.add_argument('--n-top-topics', default=20, type=int,
                        help='Number of top topics to extract for final df')

    parser.add_argument('--data-dir',
                        default=os.path.join(r'C:\Users\jjnunez\PycharmProjects', 'scar_nlp_data', 'data'))

    parser.add_argument('--results-dir', default=os.path.join(r'C:\Users\jjnunez\PycharmProjects\scar_nlp_psych', 'results'))

    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])

    args = parser.parse_args()

    return args