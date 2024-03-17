from run_bertopic.args import get_args
from run_bertopic.run_bertopic import RunBERTopic
import os.path

if __name__ == '__main__':

    print("loading arguments")

    args = get_args()
    results_dir = os.path.join(args.project_dir, "results")
    data_dir = os.path.join(args.project_dir, r"./data/processed")

    filename_to_load = f'{args.doc_types}_{args.years}.csv'
    f = os.path.join(data_dir, filename_to_load)

    print('Starting BERTopic analysis of Reddit AMA comments')

    bt = RunBERTopic(args, f, results_dir)

    bt.fit_topic_model()

    bt.extract_top_topics(args.n_top_topics)

    print('Completed BERTopic analysis of Reddit AMA comments')

