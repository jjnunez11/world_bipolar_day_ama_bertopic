import os.path
import warnings
from data_processor.args import get_args
from os import path

from data_processor.reddit_doc_processor import RedditDocProcessor

if __name__ == "__main__":
    args = get_args()
    data_dir = path.join(args.project_dir, r"./data/")

    p = RedditDocProcessor(data_dir, args)

    print(f"Finished preprocessing {args.doc_types} documents from years: {args.doc_types}")



