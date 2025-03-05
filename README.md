# world_bipolar_day_ama_bertopic

## Steps to Run:

1. Place data in `.\data\raw`. Data should be in csv format, with columns: Comment,Score,Author,Created,ParentUsername
2. Run the data processor with `python -m data_processor`. For a list of arguments that can be passed, use `python -m data_processor --help`. An example of running it on Windows is found in `.\run_proc.bat`
3. Run the BERTopic pipeline using `python -m run_bertopic`. Again, use `python -m run_bertopic --help` for a list of arguments 
4. Results of the topic modelling etc should then be available in `.\data\processed`. `.\run.bat` shows an example of the commands ran in Windows for the pipeline.