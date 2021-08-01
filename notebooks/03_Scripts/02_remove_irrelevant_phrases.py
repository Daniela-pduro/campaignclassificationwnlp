import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import CampaignsAfterEDAFileGetInfo

# Feature Engineering
from campaigns.modeling.preprocessing import IrrelevantPhrasesRemover

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add datafile')
    
    # Creating variables 'input_root', 'input_file', 'output_root', 'output_file'
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help= 'Output root')
    parser.add_argument('--output_file', type=str, help='Output file')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    print(args.input_file)
    
    # Getting the df
    # Creating an instance
    vertical_model = CampaignsAfterEDAFileGetInfo(dataroot=args.input_root, datafile=args.input_file)
    
    # Removing noisy data
    # Creating an instance
    
    print('File 01/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant.txt')
    
    print('File 02/39: ')

    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_02.txt')
    
    print('File 03/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_03.txt')
    
    print('File 04/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_04.txt')
    
    print('File 05/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_05.txt')
    
    print('File 06/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_06.txt')
    
    print('File 07/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_07.txt')
    
    print('File 08/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_08.txt')
    
    print('File 09/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_09.txt')
    
    print('File 10/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_10.txt')
    
    print('File 11/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_11.txt')
    
    print('File 12/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_12.txt')
    
    print('File 13/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_13.txt')
    
    print('File 14/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_14.txt')
    
    print('File 15/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_15.txt')
    
    print('File 16/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_16.txt')
    
    print('File 17/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_17.txt')
    
    print('File 18/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_18.txt')
    
    print('File 19/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_19.txt')
    
    print('File 20/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_20.txt')
    
    print('File 21/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_21.txt')
    
    print('File 22/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_22.txt')
    
    print('File 23/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_23.txt')
    
    print('File 24/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_24.txt')
    
    print('File 25/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_25.txt')
    
    print('File 26/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_26.txt')
    
    print('File 27/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_27.txt')
    
    print('File 28/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_28.txt')
    
    print('File 29/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_29.txt')
    
    print('File 30/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_30.txt')
    
    print('File 31/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_31.txt')
    
    print('File 32/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_32.txt')
    
    print('File 33/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_33.txt')
    
    print('File 34/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_34.txt')
    
    print('File 35/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_35.txt')
    
    print('File 36/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_36.txt')
    
    print('File 37/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_37.txt')
    
    print('File 38/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_38.txt')
    
    print('File 39/39: ')
    
    vertical_model = IrrelevantPhrasesRemover(
        vertical_model.campaigns_df, '../../../corpora/irrelevant_39.txt')
                        
    # Saving file
    
    vertical_model.save_final_df(args.output_root, args.output_file)
   