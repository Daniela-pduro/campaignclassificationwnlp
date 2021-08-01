import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import LanguageFileGetInfo

# Texts Preprocessing
from campaigns.modeling.nlp import VerticalMessagePreprocessor


if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add language file arguments')
    
    # Creating variables
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help='Output root')
    parser.add_argument('--output_file', type=str, help='Output file')
    parser.add_argument('--stemmer', type=str, help='Stemmer Language')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    
    # Getting the df
    # Creating an instance
    language_instance = LanguageFileGetInfo(dataroot=args.input_root, datafile=args.input_file)
    
    print('Language df shape before cleaning texts:', language_instance.language_df.shape)

    
    # Preprocessing Texts
    
    # Creating an instance
    language_instance = VerticalMessagePreprocessor(
        language_instance.language_df, args.stemmer, remove_unlabeled=False)
    
    # Preprocessing
    language_instance.preprocess_message(flag_stemming=True)
    print('Language df shape after cleaning texts:', language_instance.language_df.shape)
    
    # Saving the file
    language_instance.save_final_df(args.output_root, args.output_file)    