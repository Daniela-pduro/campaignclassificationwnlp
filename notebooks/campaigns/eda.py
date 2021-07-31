# Directories Handle
import os


# Data Processing
import numpy as np
import pandas as pd
from scipy.stats import skew
import scipy.stats as stats

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Getting random samples
import random

class ExploratoryDataAnalyser(object):
    """
    
    This python class reads the tabular data contained in a dataframe and:
    
    - Displays a summary: dimensions, data types, duplicated values, missing texts.
    - Displays a bar chart with the relative frequencies of campaigns by customer category.
    - Gets main numerical variables `total_sent`, `opens` and `clicks` and displays how data is spread.
    - Plots the distribution of the main numerical variables `total_sent`, `opens` and `clicks`.
    - Normalizes main numerical variables `total_sent`, `opens` and `clicks`.
    - Plots outliers of `total_sent`, `opens` and `clicks`.
    - Detects outliers of the main numerical variables.
    - Handles the outliers of the main numerical variables.
    - Performs a bivariate analysis of by customer category of the main numerical variables.
    """
    
    # Initializer
    def __init__(self, campaigns_df):
        
        self.campaigns_df = campaigns_df
        
    # Instance Methods
    
    def pre_eda(self):
        """
        - Displays dataframe dimensions.
        - Displays data types.
        - Displays duplicated values, if any.
        - Displays missing texts, if any.
        
        """
        print(f'\nNumber of observations:', self.campaigns_df.shape[0])
        print(f'\nNumber of columns: {self.campaigns_df.shape[1]}')
        print('\nCampaigns Info.:')
        print('======================================================')
        print(self.campaigns_df.info())
        print('\nUnique elements per variable:')
        print('======================================================')
        print(self.campaigns_df.nunique())
        print('\nNumber of duplicated observations:',len(self.campaigns_df)-len(self.campaigns_df.drop_duplicates()))
        print('Number of campaigns without subject: ', self.campaigns_df['subject']\
              .isnull().sum())
        print('Number of campaigns without message: ', self.campaigns_df['message']\
              .isnull().sum())
    
    def customer_cat_to_cat_converter(self):
        """
        - Converts 'customer_cat' column into categorical.
        """
        
        self.campaigns_df['customer_cat'] = pd.Categorical(self.campaigns_df['customer_cat'])
        
        print(self.campaigns_df.dtypes)
  
    def cats_plotter(self):
        """
        - Gets main categorical variable ('customer_cat').
        - Displays a bar chart with the relative frequencies of campaigns by customer category.
        """

        ## plotting and adding annotations
        
        active_accounts = (self.campaigns_df['customer_cat'].value_counts(normalize=True) * 100).round(decimals=2)
        
        sns.set_style("darkgrid")
        
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle('Percentage of Campaigns by Customer Type', fontsize=16)
        
        bars = ax.bar(active_accounts.index,active_accounts.values, 
                      color='#56B4E9', alpha=0.80 )
        
        plt.xlabel('Category', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        
        for bar in bars:
            height = bar.get_height()
            label = bar.get_x() + bar.get_width() / 2
            ax.text(label, height, s=f'{height}%', ha='center',
                   va='bottom')
            
        plt.show()
                    
    def get_correlation_matrix(self):
        """
        - Displays a corrlation matrix of the main numerica variables in campaigns.
        """
        
        corr_df = self.campaigns_df[['total_sent', 'opens', 'clicks']]

        corr_matrix = corr_df.corr()
        sns.heatmap(corr_matrix , annot=True)
        plt.show()
    
    def describe_overall(self):
        """
        - Gets main numerical variables.
        - Displays how data is spread (percentiles).
        - Displays main numerical variables variance.
       
        """ 
        overall_stats = self.campaigns_df[['total_sent', 'opens', 'clicks']].describe()
        
        overall_variance = self.campaigns_df[['total_sent', 'opens', 'clicks']].var()
        
        print('Overall Variance:\n')
        print(overall_variance)
        
        return overall_stats
    
    def plot_main_var(self):
        """
        - Plots main numerical variables distributions.
        - Displays skewness.
        
        """
        sns.set_style("darkgrid")
        
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle("Main Variables Distribution before Normalization", fontsize=16)
        
        total_sent = self.campaigns_df['total_sent']
        opens = self.campaigns_df['opens']
        clicks = self.campaigns_df['clicks']
        
        plt.hist(total_sent, bins=15, color='#E69F00', label='Total Sent', alpha=0.75)
        plt.hist(opens, bins=15, color='#56B4E9', label='Opens', alpha=0.80)
        plt.hist(clicks, bins=15, color='#2CA02C', label='Clicks', alpha=0.75)
        
        plt.legend()
        
        print(f'Total Sent Skweness: {skew(self.campaigns_df.total_sent):.2f}')
        print(f'Opens Skweness: {skew(self.campaigns_df.opens):.2f}')
        print(f'Clicks Skewness: {skew(self.campaigns_df.clicks):.2f}')
    
    
    def describe_by_cat(self, variable):
        """
        - Displays how the data of a given variable passed as input to the function 
        is spread (percentiles) by customer category.
        - Displays the variance of the variable.
        """
        
        stats = self.campaigns_df.groupby('customer_cat')[variable].describe()
        variance = self.campaigns_df.groupby('customer_cat')[variable].var()
        
        print('Variance:\n')
        print(variance)
        
        return stats


    def log_normalize(self):
        """
         - Normalizes main numerical variables
        
        """
        
        # Normalizing
        self.campaigns_df[['total_sent', 'opens', 'clicks']] = self.campaigns_df[
            ['total_sent', 'opens', 'clicks']]\
            .applymap(lambda x: np.log10(x + 1))
        
        # Plotting Histograms
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle("Main Variables Distribution After Normalization",fontsize=16)
        
        
        total_sent = self.campaigns_df['total_sent']
        opens = self.campaigns_df['opens']
        clicks = self.campaigns_df['clicks']
        
        ax.hist(total_sent, bins=15, color='#E69F00', label='Total Sent', alpha=0.95)
        ax.hist(opens, bins=15, color='#56B4E9', label='Opens', alpha=0.70)
        ax.hist(clicks, bins=15, color='#2CA02C', label='Clicks', alpha=0.50)
        
        plt.legend()
        
        print(f'Total Sent Skweness: {skew(self.campaigns_df.total_sent):.2f}')
        print(f'Opens Skweness: {skew(self.campaigns_df.opens):.2f}')
        print(f'Clicks Skewness: {skew(self.campaigns_df.clicks):.2f}')
   
    
    def plot_outliers(self):
        """
        - Displays Boxplots to identify outliers.
        """
        
        # Plotting box plots
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle("Main Variables Boxplots - Outliers Detection", fontsize=16)
        
        total_sent = self.campaigns_df['total_sent']
        opens = self.campaigns_df['opens']
        clicks = self.campaigns_df['clicks']
        
        colors = ['#E69F00','#56B4E9','#2CA02C']
        colors_total_sent = dict(color=colors[0])
        colors_opens = dict(color=colors[1])
        colors_clicks = dict(color=colors[2])
        labels = ['Total Sent','Open Emails', 'Total Clicks']
        
        ax.boxplot(total_sent, positions=[0],labels=[labels[0]], 
                   boxprops=colors_total_sent, medianprops=colors_total_sent,
                   whiskerprops=colors_total_sent, capprops=colors_total_sent,
                   flierprops=dict(markeredgecolor=colors[0]))
        
        ax.boxplot(opens, positions=[1], labels=[labels[1]],
                   boxprops=colors_opens, medianprops=colors_opens,
                   whiskerprops=colors_opens, capprops=colors_opens,
                   flierprops=dict(markeredgecolor=colors[1]))
        
        ax.boxplot(clicks, positions=[2], labels=[labels[2]],
                   boxprops=colors_clicks, medianprops=colors_clicks,
                   whiskerprops=colors_clicks, capprops=colors_clicks,
                   flierprops=dict(markeredgecolor=colors[2]))
    
    
    def detect_outliers(self, variable):
        """
        - Displays outliers from a given variable passed as argument to the function.
        - Returns fence values.
        """
        
        q1 = self.campaigns_df[variable].quantile(0.25)
        q3 = self.campaigns_df[variable].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print(f'\nOutliers:')
        print(len(self.campaigns_df[(self.campaigns_df[variable] < lower_bound)|
                                 (self.campaigns_df[variable] > upper_bound)]))
        
        
        return lower_bound, upper_bound
        
    
    def handle_outliers(self, variable, lower_bound, upper_bound):
        """
        - Receives 'total_sent' fence values: 
        total_sent_lower_bound, total_sent_upper_bound.
        - Substitutes outliers by fence values.
        - Displays mean value after handling outliers.
        """

        self.campaigns_df.loc[(self.campaigns_df[variable] < lower_bound), variable] = lower_bound
        self.campaigns_df.loc[(self.campaigns_df[variable] > upper_bound), variable] = upper_bound

        print(f'Mean value after outliers treatment: {self.campaigns_df[variable].mean():.2f}')
        
    
    def bivariate_plotter(self, variable):
        """
        - Performs a bivariate analysis - Customer Cat vs. Total Sent
        - Displays a box plot
        - Displays a bar chart.
        """
        
        # Plotting box plots
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle(str(variable) + ' by customer category', fontsize=16)
                
        colors_dict = {'total_sent': '#E69F00', 'opens':'#56B4E9', 'clicks': '#2CA02C'}
        
        color = [colors_dict[variable]]
        colors_customer = dict(color=color[0])
        labels = ['Category 01','Category 02', 'Category 03', 'Category 04',
                 'Category 05']
        
        for customer in range(1,6):
        
            ax.boxplot(self.campaigns_df[self.campaigns_df['customer_cat']==customer][variable], 
                       positions= [customer-1], labels=[customer], 
                       boxprops=colors_customer, 
                       medianprops=colors_customer,
                       whiskerprops=colors_customer, 
                       capprops=colors_customer,
                       flierprops=dict(markeredgecolor=color[0]))
        
        # Plotting bar charts
        for customer in range(1,6):
            
            fig, ax = plt.subplots(1, figsize=(8,5))
            plt.suptitle('Category ' + str(customer), fontsize=16)
            
            ax.hist(self.campaigns_df[self.campaigns_df['customer_cat']==customer][variable], 
                    bins=15, color=colors_dict[variable], label=variable, alpha=0.75)
            
    def random_sampler(self, number_of_samples, random_seed):
        """
        - Receives the number of elements in the sample and the random seed as input arguments.
        - Generates a randomized sample.
        - Groups the sample by customer category.
        - Returns the randomized sample and the grouped dataframes.
        """
        
        customer_cat = list(self.campaigns_df['campaign_id'].unique())
        
        random.seed(random_seed)
        
        campaign_id_sample = random.sample(customer_cat, number_of_samples)
        
        campaign_sample = self.campaigns_df[
            self.campaigns_df['campaign_id'].isin(campaign_id_sample)].reset_index(drop=True)
        
        campaign_sample = campaign_sample[['customer_cat', 'total_sent', 'opens', 'clicks']]
        
        groups = campaign_sample.groupby('customer_cat').count().reset_index()
        
        return campaign_sample, groups

    def qq_plotter(self, campaign_sample, variable):
        """
        - Receives a sample and the name of a variable as input.
        - Returns a qq plot to check normality of the variable passed as an argument to the function.
        """
        unique_cats = campaign_sample['customer_cat'].unique()

        for category in unique_cats:
            
            stats.probplot(campaign_sample[
                campaign_sample['customer_cat']==category][variable], dist='norm', plot=plt)
        
            plt.show()

    def homog_var_checker(self, campaign_sample):
        """
        - Receives a sample as input argument.
        - Checks homogeinity of variance.
        - Returns ratio of the largest to the smallest group.
        """
        
        ratio = campaign_sample.groupby(
            'customer_cat').std().max() / campaign_sample.groupby('customer_cat').std().min()

        return ratio
            
    def anova_tester(self, campaign_sample, variable, alpha, tail_hypothesis):
        """
        - Receives a samples as input argument.
        - Performs anova test.
        """
        
        # Creating table
        data = [['Between Groups',  '', '', '', '', '', ''],[
            'Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
        
        anova_table = pd.DataFrame(data, columns = [
            'Source of Variation', 'sum_of_squares', 'degrees_of_freedom', 'MS', 'F_val', 'P_value',
            'F_crit'])
        
        anova_table.set_index('Source of Variation', inplace=True)
        
        x_bar = campaign_sample[variable].mean()
        
        # Sum of Squares
        sstr = campaign_sample.groupby(
            'customer_cat').count() * (campaign_sample.groupby('customer_cat').mean() - x_bar)**2
        
        anova_table['sum_of_squares']['Between Groups'] = sstr[variable].sum()
        
        sse = (campaign_sample.groupby(
            'customer_cat').count() - 1) * campaign_sample.groupby('customer_cat').std() ** 2
        
        anova_table['sum_of_squares']['Within Groups'] = sse[variable].sum()
        
        ssto = sstr[variable].sum() + sse[variable].sum()
        
        anova_table['sum_of_squares']['Total'] = ssto
        
        # degrees of freedom
        
        anova_table['degrees_of_freedom']['Between Groups'] = campaign_sample['customer_cat'].nunique() - 1
        anova_table['degrees_of_freedom']['Within Groups'] = campaign_sample.shape[0] - campaign_sample[
            'customer_cat'].nunique()
        anova_table['degrees_of_freedom']['Total'] = campaign_sample.shape[0] - 1
        
        # MS
        anova_table['MS'] = anova_table['sum_of_squares'] / anova_table['degrees_of_freedom']
        
        # F stat
        f_val = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
        anova_table['F_val']['Between Groups'] = f_val
        
        # p-value

        anova_table['P_value']['Between Groups'] = 1 - stats.f.cdf(f_val, anova_table[
            'degrees_of_freedom']['Between Groups'], anova_table['degrees_of_freedom']['Within Groups'])
        
        # choosing a significance level

        alpha = 0.05
        
        # choosing type of distribution
        
        if tail_hypothesis == "one_tailed":
            
            anova_table['F_crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table[
                'degrees_of_freedom']['Between Groups'], anova_table[
                'degrees_of_freedom']['Within Groups'])
        
        if tail_hypothesis == "two_tailed":
            
            alpha /= 2

            anova_table['F_crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table[
                'degrees_of_freedom']['Between Groups'], anova_table[
                'degrees_of_freedom']['Within Groups'])
        

        print("F Value: ", anova_table['F_val']['Between Groups'])
        print("p-value: ", anova_table['P_value']['Between Groups']) 
        print("Critical f-value :", anova_table['F_crit']['Between Groups'])

        if anova_table['F_crit']['Between Groups'] > anova_table['F_val']['Between Groups'] or anova_table['P_value']['Between Groups'] <= alpha:
            
            print("The difference between some of the means are statistically significant.")
            
            print("The null hypothesis is rejected; not all of the population means are equal.")        
        
        return anova_table
    
    def save_final_df(self, path, filename):
        """
        -  Saves the df in the specified path.
        """
        self.campaigns_df.to_csv(os.path.join(path,filename), header=None)
            
class NLPExploratoryDataAnalyser(object):
    """
    This python class reads the tabular data contained in a specific language file and:
    
    - Creates a new dataframe with the columns 'message' and 'sector'.
    -
    
    """
    # Initializer
    def __init__(self, language_df):
        self.language_df = language_df
        self.nlp_df = self.language_df[['message', 'sector']]
    
    # Instance Methods