import os.path
import warnings
from data_processor.args import get_args
from os import path
import pandas as pd

from data_processor.reddit_doc_processor import RedditDocProcessor

if __name__ == "__main__":
    args = get_args()
    data_dir = path.join(args.project_dir, r"./data/")

    raw_data_dir = os.path.join(data_dir, "raw/")
    processed_data_dir = os.path.join(data_dir, "processed/")

    if args.years == "all":

        years = ["2019", "2020", "2021", "2022", "2023"]
        for year in years:
            args.years = year
            p = RedditDocProcessor(args, raw_data_dir, processed_data_dir)
            p.process_docs()

        merged_df = pd.DataFrame()
        for filename in os.listdir(processed_data_dir):
            if filename.startswith(args.doc_types) and filename.endswith(".csv") and "all" not in filename:

                # Read the CSV file into a DataFrame
                df = pd.read_csv(os.path.join(processed_data_dir, filename), index_col=None)
                df.columns = ['comments']
                print(f'Merging in {filename}, with {df.shape[0]} rows, and size: {df.shape} ')
                # Append the DataFrame to the list
                merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)
                # print(f'Df to merge is now length: {dfs.shape[0]}')

        # Concatenate all DataFrames into a single DataFrame
        # merged_df = pd.concat(dfs, ignore_index=True, axis=0, sort=False)

        # Write the merged DataFrame to a new CSV file
        output_filename = f"{args.doc_types}_all.csv"
        output_path = os.path.join(processed_data_dir, output_filename)
        merged_df.to_csv(output_path, index=False, header=False)

    else:
        p = RedditDocProcessor(args, raw_data_dir, processed_data_dir)
        p.process_docs()

    print(f"Finished preprocessing {args.doc_types} documents from years: {args.doc_types}")



