import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import ClassifiedCampaignsFileGetInfo

# Feature Engineering
from campaigns.modeling.preprocessing import BenchmarkCreator, Benchmarker

# Texts Preprocessing
from campaigns.modeling.nlp import MainModelTextsPreprocessor

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add language argument')
    
    # Creating variables
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help='Output root')
    parser.add_argument('--output_file', type=str, help='Output file')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    
    # Getting the df
    # Creating an instance
    campaigns = ClassifiedCampaignsFileGetInfo(args.input_root, args.input_file)
    
    # Creating new variables - Benchmarks by customer_cat and vertical
    # Creating an instance
    campaigns = BenchmarkCreator(campaigns.campaigns_w_vertical_df)
    
    # Creating auxiliary df containing benchmarks by customer_cat and vertical
    aux = campaigns.benchmarks_aux_creator()
    
    # Creating new variables - benchmarks (open_rate_benchmark, ctr_benchmark, ctor_benchmark)
    campaigns.benchmark_variable_creator(aux)
    
    
    # Creating new variables - target variables 
    # Creating an instance
    campaigns = Benchmarker(campaigns.campaigns_w_vertical_df)
    
    # Creating new variables: 'open_rate_result', 'ctr_result', 'ctor_result'
    campaigns.campaign_benchmarker('open_rate', 'open_rate_benchmark')
    campaigns.campaign_benchmarker('ctr','ctr_benchmark')
    campaigns.campaign_benchmarker('ctor','ctor_benchmark')
    
    # Preparing features and target
    # Creating and instance
    campaigns = MainModelTextsPreprocessor(campaigns.campaigns_w_vertical_df, 'spanish', remove_unlabeled=False)
    
    # Preprocessing texts
    campaigns.preprocess_message(flag_stemming=True)
    
    # Show df columns
    print('Campaigns df columns: ', campaigns.campaigns_w_vertical_df.columns)
    
    # Saving file
    campaigns.save_final_df(args.output_root, args.output_file)