# Directories Handle
import os


# Warnings
import warnings
warnings.filterwarnings('ignore')

# Data Processing
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_rows = 100

class CampaignsFileGetInfo(object):
    """
    This python class reads the tabular data contained in a CSV file and:
    
    - Drops inactive accounts.
    - Adds a column with campaign's ID for easier reference.
    
    
    """
    # Initializer    
    def __init__(self, dataroot, datafile):
        
        self.dataroot = dataroot
        self.datafile = os.path.join(dataroot,datafile)
        self.campaigns_df = self.get_input()
        self.blocked = self.campaigns_df[self.campaigns_df['customer_cat'] == -1]
        self.campaigns_df.drop(self.blocked.index, inplace=True)

    # Instance Methods   
    def get_input(self):
        """
        Get campaign's data sample:
        
        - Adds a column with campaign's ID for easier reference
        
        :returns: pandas dataframe
        
        """
        campaigns_df = pd.read_csv(self.datafile, sep='^', 
                                   parse_dates=['date_sent'], encoding='utf-8')
        
        pd.options.display.float_format = '{:,.2f}'.format
        
        campaigns_df['campaign_id'] = campaigns_df.index
        
        return campaigns_df
    
class CampaignsEDAFileGetInfo(object):
    """
    This python class reads the tabular data contained in the folder output_01 and returns a pandas dataframe.
    """
    
    # Initializer
    def __init__(self, dataroot, datafile):
        
        self.dataroot = dataroot
        self.datafile = self.dataroot + datafile
        self.campaigns_df = self.get_input()
        
    # Instance methods
    def get_input(self):
        """
        Get language's data sample:

        - Returns a pandas dataframe.
        """
    
        columns = ['sender', 'subject', 'date_sent', 'total_sent',
                   'customer_cat', 'opens','clicks', 'sector', 'message', 'campaign_id']
    
        campaigns_df = pd.read_csv(self.datafile, names=columns)
    
        pd.options.display.float_format = '{:,.2f}'.format
    
        return campaigns_df     

class CampaignsAfterEDAFileGetInfo(object):
    """
    This python class reads the tabular data contained in a csv file and returns a pandas dataframe.
    """
    
    # Initializer
    def __init__(self, dataroot, datafile):
        
        self.dataroot = dataroot
        self.datafile = self.dataroot + datafile
        self.campaigns_df = self.get_input()
        
    # Instance methods
    def get_input(self):
        """
        Get language's data sample:

        - Returns a pandas dataframe.
        """
    
        columns = ['sender', 'subject', 'date_sent', 'total_sent', 'customer_cat', 'opens', 'clicks',
                   'sector', 'message', 'campaign_id', 'open_rate', 'ctr', 'ctor', 'clean_subject', 'clean_message']
    
        campaigns_df = pd.read_csv(self.datafile, names=columns)
    
        pd.options.display.float_format = '{:,.2f}'.format
    
        return campaigns_df     

    
class LanguageFileGetInfo(object):
    """
    This python class reads the tabular data contained in a dataframe and returns a pandas dataframe.   
    """
    
    # Initializer
    def __init__(self, dataroot, datafile):
        
        self.dataroot = dataroot
        self.datafile = os.path.join(dataroot,datafile)
        self.language_df = self.get_input()
        
    # Instance Methods 
    def get_input(self):
        """
        Get language's data sample:
        
        - Returns a pandas dataframe.
        
        """
        columns = ['sender', 'subject', 'date_sent', 'total_sent', 'customer_cat', 'opens', 'clicks', 
                   'sector', 'message', 'campaign_id', 'open_rate', 'ctr', 'ctor', 'clean_subject',
                   'clean_message']
        
        language_df = pd.read_csv(self.datafile, names=columns)
        
        pd.options.display.float_format = '{:,.2f}'.format
        
        return language_df
    

class CleanCampaignsFileGetInfo(object):
    """
    This python class reads the tabular data contained in a CSV file and returns a pandas dataframe.
    """
    
    # Initializer
    
    def __init__(self, data_root, datafile):
        
        self.data_root = data_root
        self.datafile = os.path.join(data_root,datafile)
        self.language_df = self.get_input()
        
    # Instance Methods
    
    def get_input(self):
        """
        - Gets vertical_classified_df and returns a pandas dataframe.
        """
        
        columns = ['sender', 'subject', 'date_sent', 'total_sent', 'customer_cat', 'opens', 'clicks', 'sector', 'message',
                   'campaign_id', 'open_rate', 'ctr', 'ctor', 'clean_subject', 'clean_message']
        
        language_df = pd.read_csv(self.datafile, lineterminator='\n', header=None, names=columns)
                                           
        pd.options.display.float_format = '{:,.2f}'.format
        
        return language_df

    
class ClassifiedCampaignsFileGetInfo(object):
    """
    This python class reads the tabular data contained in a CSV file and returns a pandas dataframe.
    """
    
    # Initializer
    
    def __init__(self, data_root, datafile):
        
        self.data_root = data_root
        self.datafile = os.path.join(data_root,datafile)
        self.campaigns_w_vertical_df = self.get_input()
        
    # Instance Methods
    
    def get_input(self):
        """
        - Gets vertical_classified_df and returns a pandas dataframe.
        """
        
        columns = ['sender', 'subject', 'date_sent', 'total_sent', 'customer_cat', 'opens', 'clicks', 'message', 'campaign_id',
                   'open_rate', 'ctr', 'ctor', 'clean_subject', 'clean_message', 'vertical']
        
        campaigns_w_vertical_df = pd.read_csv(self.datafile, lineterminator='\n', header=None, names=columns)
                                           
        pd.options.display.float_format = '{:,.2f}'.format
        
        return campaigns_w_vertical_df


class CleanClassifiedCampaignsFileGetInfo(object):
    """
    This python class reads the tabular data contained in a CSV file and returns a pandas dataframe.
    """
    
    # Initializer
    
    def __init__(self, data_root, datafile):
        
        self.data_root = data_root
        self.datafile = os.path.join(data_root,datafile)
        self.campaigns_w_vertical_df = self.get_input()
        
    # Instance Methods
    
    def get_input(self):
        """
        - Gets vertical_classified_df and returns a pandas dataframe.
        """
        
        columns = ['sender', 'subject', 'date_sent', 'total_sent', 'customer_cat', 'opens', 'clicks',
                   'message', 'campaign_id', 'open_rate', 'ctr', 'ctor', 'clean_subject', 'clean_message',
                   'vertical', 'open_rate_benchmark', 'ctr_benchmark', 'ctor_benchmark', 'open_rate_result',
                   'ctr_result', 'ctor_result']
        
        campaigns_w_vertical_df = pd.read_csv(self.datafile, lineterminator='\n', header=None, names=columns)
                                           
        pd.options.display.float_format = '{:,.2f}'.format
        
        return campaigns_w_vertical_df


