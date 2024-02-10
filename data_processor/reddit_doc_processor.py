import os
import pandas as pd

from data_processor.globals import FILENAME_DICT


class RedditDocProcessor(object):

    def __init__(self, args, data_dir):

        self.raw_data_dir = os.path.join(data_dir, "raw/")
        self.processed_data_dir = os.path.join(data_dir, "processed/")

        if args.years == "all":
            raise NotImplementedError
            self.f = None
            self.data = None
        else:
            self.f = os.path.join(self.raw_data_dir, FILENAME_DICT[args.years])
            self.data = pd.read_csv(self.f)

        self.type = args.doc_types
        self.years = args.years

    def process_docs(self):

        # Load data from the file

        if self.type == "participant_posts":
            self.process_participant_posts()
        elif self.type == "team_comments":
            raise NotImplementedError
        elif self.type == "participant_comments":
            raise NotImplementedError
        elif self.type == "all":
            raise NotImplementedError
        else:
            raise ValueError

    def process_participant_posts(self):

        # Filter rows to only include parent posts that weren't made by the mod bot or the CREST_BD team
        filtered_data = self.data[(self.data['ParentUsername'].isna()) &
                                  (self.data['Author'] != 'IAmAModBot') &
                                  (self.data['Author'] != 'CREST_BD')]

        # Remove quotation marks from the text in the Comment column
        filtered_data['Comment'] = filtered_data['Comment'].str.replace('"', '')

        # Keep only the Comment column, but not the header
        filtered_data = filtered_data["Comment"]

        # Construct output filename
        output_filename = f"{self.type}_comments_{self.years}.csv"
        output_path = os.path.join(self.processed_data_dir, output_filename)

        # Write filtered data to CSV
        filtered_data.to_csv(output_path, index=False, header=False)

        print(f"Filtered data saved to: {output_path}")
