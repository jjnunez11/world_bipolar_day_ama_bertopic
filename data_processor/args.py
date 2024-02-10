import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="CAN-BIND-1 Data Processing")

    # Files and directories
    parser.add_argument('--project-dir', default="./", help="Path to root of project")
    parser.add_argument('--years', default="2023", choices=["2019", "2020", "2021", "2022", "2023", "all"])
    parser.add_argument('--doc_types', default="participant_posts", choices=["participant_posts", "team_comments",
                                                                             "participant_comments", "all"])
    args = parser.parse_args()
    return args
