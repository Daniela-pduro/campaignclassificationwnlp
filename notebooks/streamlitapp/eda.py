# Directories Handle
import os

import pandas as pd
import numpy as np

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import streamlit as st

class ExploratoryDataAnalyser(object):
    """
    
    This python class reads the tabular data contained in a dataframe and:
    
    - Normalizes main numerical variables.
    - Plots boxplots and histograms combinining customer categories and the main variables: 'total_sent', 'opens' and 'clicks'.
    """
    
    # Initializer
    
    def __init__(self, campaigns_df):
        
        self.campaigns_df = campaigns_df
        self.fillcolor_palette = {'overview': 'aliceblue', '1':'rgb(146, 242, 216)', '2':'rgb(226, 140, 121)',
                                  '3':'rgb(104, 205, 223)','4':'rgb(238, 150, 201)','5':'rgb(252, 240,204)'}
        
        self.marker_palette = {'overview': 'rgb(157, 141, 241)', '1':'rgb(24, 201, 154)', 
                               '2':'rgb(218, 112, 88)','3':'rgb(53, 188, 212)', '4':'rgb(193, 31, 126)',
                               '5':'rgb(243, 193, 43)'}
        
    # Instance methods   
    def log_normalize(self):
        """
        - Normalizes main numerical variables.
        """
        
        # Normalizing the variables 'total_sent', 'opens' and 'clicks'
        
        self.campaigns_df[['total_sent', 'opens', 'clicks']] = self.campaigns_df[
            ['total_sent', 'opens', 'clicks']]\
            .applymap(lambda x: np.log10(x + 1))
    
    
    def remove_total_sent_outliers(self):
        """
        - Detects 'total_sent' fence values: total_sent_lower_bound and total_sent_upper_bound.
        - Substitutes outliers  by fence values.
        """
        
        total_sent_q1 = self.campaigns_df['total_sent'].quantile(0.25)
        total_sent_q3 = self.campaigns_df['total_sent'].quantile(0.75)
        total_sent_iqr = total_sent_q3 - total_sent_q1
        
        total_sent_lower_bound = total_sent_q1 - 1.5 * total_sent_iqr
        total_sent_upper_bound = total_sent_q3 + 1.5 * total_sent_iqr
        
        self.campaigns_df.loc[(self.campaigns_df['total_sent'] < total_sent_lower_bound), 'total_sent'] = total_sent_lower_bound
        self.campaigns_df.loc[(self.campaigns_df['total_sent'] > total_sent_upper_bound), 'total_sent'] = total_sent_upper_bound
    
    def remove_open_outliers(self):
        """
        - Detects 'opens' fence values: opens_lower_bound, opens_upper_bound.
        - Substitutes outliers by fence values.
        """
        
        opens_q1 = self.campaigns_df['opens'].quantile(0.25)
        opens_q3 = self.campaigns_df['opens'].quantile(0.75)
        opens_iqr = opens_q3 - opens_q1
        
        opens_lower_bound = opens_q1 - 1.5 * opens_iqr
        opens_upper_bound = opens_q3 + 1.5 * opens_iqr
        
        self.campaigns_df.loc[(self.campaigns_df['opens'] < opens_lower_bound), 'opens'] = opens_lower_bound
        self.campaigns_df.loc[(self.campaigns_df['opens'] > opens_upper_bound), 'opens'] = opens_upper_bound
        
    def remove_clicks_outliers(self):
        """
        - Detects 'clicks' fence values: clicks_lower_bound, clicks_upper_bound.
        - Substitutes outliers by fence values.
        """
        clicks_q1 = self.campaigns_df['clicks'].quantile(0.25)
        clicks_q3 = self.campaigns_df['clicks'].quantile(0.75)
        clicks_iqr = clicks_q3 - clicks_q1
        
        clicks_lower_bound = clicks_q1 - 1.5 * clicks_iqr
        clicks_upper_bound = clicks_q3 + 1.5 * clicks_iqr
        
        self.campaigns_df.loc[(self.campaigns_df['clicks'] < clicks_lower_bound), 'clicks'] = clicks_lower_bound
        self.campaigns_df.loc[(self.campaigns_df['clicks'] > clicks_upper_bound), 'clicks'] = clicks_upper_bound
        
    def display_overview(self):
        """
        - Displays a box_plot and histogram per category for variables 'total_sent', 'opens' and 'clicks'.
        """
        
        variable_list = ['total_sent', 'opens', 'clicks']
        
        for variable in variable_list:
            
            # Figure setup
            
            fig = make_subplots(rows=12, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.02)
    
            # Traces
            x = self.campaigns_df[variable]
    
            fig.add_trace(go.Box(x=x,name='All categories', boxmean=True, 
                                 fillcolor=self.fillcolor_palette['overview'], 
                                 marker=dict(color=self.marker_palette['overview'])), row=1,col=1)

            fig.add_trace(go.Histogram(x=x, marker=dict(color=self.marker_palette['overview'])),
                          row=2, col=1)
    
            row = 2
        
        
            for category in range(1,6):
        
                data = self.campaigns_df[self.campaigns_df['customer_cat'] == category][variable]
        
                row += 1
        
                fig.add_trace(go.Box(x=data,name='category ' + str(category), boxmean=True,
                                     fillcolor=self.fillcolor_palette[str(category)],
                                     marker=dict(color=self.marker_palette[str(category)])), row= row,
                              col=1)
        
                row += 1
        
                fig.add_trace(go.Histogram(x=data,marker=dict(
                                color=self.marker_palette[str(category)])), 
                              row=row, col=1)
            
            # Make adjustements
            fig.update_layout(height=2000, width=800, title_text=variable, showlegend = False)

            st.plotly_chart(fig)
            
    def display_bycustomercat_total_sent(self):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' by customer category.
        """
        
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        x = self.campaigns_df['total_sent']
        
        fig.add_trace(go.Box(x=x, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['overview'],           marker=dict(color=self.marker_palette['overview'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=x, marker=dict(color=self.marker_palette['overview'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="All categories", showlegend = False)
        
        st.plotly_chart(fig)
            
    
    def display_bycustomercat_open(self):
        """
        - Displays a boxplot and a histogram of the variable 'opens' by customer category.
        """
        
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        x = self.campaigns_df['opens']
        
        fig.add_trace(go.Box(x=x, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['overview'],           marker=dict(color=self.marker_palette['overview'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=x, marker=dict(color=self.marker_palette['overview'])), row=2, col=1)
       
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="All categories", showlegend = False)
        
        st.plotly_chart(fig)

    
    def display_bycustomercat_clicks(self):
        """
        - Displays a boxplot and a histogram of the variable 'clicks' by customer category.
        """
        
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        x = self.campaigns_df['clicks']
        
        fig.add_trace(go.Box(x=x, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['overview'],           marker=dict(color=self.marker_palette['overview'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=x, marker=dict(color=self.marker_palette['overview'])), row=2, col=1)
       
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="All categories", showlegend = False)
        
        st.plotly_chart(fig)
        
    
    def display_total_sent_bycustomercat(self, customer_cat):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' and customer category 1.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == customer_cat]['total_sent']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette[str(customer_cat)],           marker=dict(color=self.marker_palette[str(customer_cat)])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette[str(customer_cat)])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category " + str(customer_cat), showlegend = False)
        
        st.plotly_chart(fig)
    
    
    
    
    
    def display_bycustomercat_total_sent_01(self):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' and customer category 1.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 1]['total_sent']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['1'],           marker=dict(color=self.marker_palette['1'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['1'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 01", showlegend = False)
        
        st.plotly_chart(fig)
        
    
    def display_bycustomercat_total_sent_02(self):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' and customer category 2.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 2]['total_sent']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['2'],           marker=dict(color=self.marker_palette['2'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['2'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 02", showlegend = False)
        
        st.plotly_chart(fig)
        
    
    def display_bycustomercat_total_sent_03(self):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' and customer category 3.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 3]['total_sent']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['3'],           marker=dict(color=self.marker_palette['3'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['3'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 03", showlegend = False)
        
        st.plotly_chart(fig)
        
    
    def display_bycustomercat_total_sent_04(self):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' and customer category 4.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 4]['total_sent']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['4'],           marker=dict(color=self.marker_palette['4'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['4'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 04", showlegend = False)
        
        st.plotly_chart(fig)
        
    
    def display_bycustomercat_total_sent_05(self):
        """
        - Displays a boxplot and a histogram of the variable 'total_sent' and customer category 5.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 5]['total_sent']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['5'],           marker=dict(color=self.marker_palette['5'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['5'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 05", showlegend = False)
        
        st.plotly_chart(fig)       
    
    def display_bycustomercat_open_01(self):
        """
        - Displays a boxplot and a histogram of the variable 'opens' and customer category 1.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 1]['opens']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['1'],           marker=dict(color=self.marker_palette['1'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['1'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 01", showlegend = False)
        
        st.plotly_chart(fig)   
        
    
    def display_bycustomercat_open_02(self):
        """
        - Displays a boxplot and a histogram of the variable 'opens' and customer category 2.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 2]['opens']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['2'],           marker=dict(color=self.marker_palette['2'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['2'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 02", showlegend = False)
        
        st.plotly_chart(fig)   
    
    def display_bycustomercat_open_03(self):
        """
        - Displays a boxplot and a histogram of the variable 'opens' and customer category 3.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 3]['opens']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['3'],           marker=dict(color=self.marker_palette['3'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['3'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 03", showlegend = False)
        
        st.plotly_chart(fig)       
    
    def display_bycustomercat_open_04(self):
        """
        - Displays a boxplot and a histogram of the variable 'opens' and customer category 4.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 4]['opens']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['4'],           marker=dict(color=self.marker_palette['4'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['4'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 04", showlegend = False)
        
        st.plotly_chart(fig)       
      
    def display_bycustomercat_open_05(self):
        """
        - Displays a boxplot and a histogram of the variable 'opens' and customer category 5.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 5]['opens']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['5'],           marker=dict(color=self.marker_palette['5'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['5'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 05", showlegend = False)
        
        st.plotly_chart(fig)       
    
    def display_bycustomercat_clicks_01(self):
        """
        - Displays a boxplot and a histogram of the variable 'clicks' and customer category 1.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 1]['clicks']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['1'],           marker=dict(color=self.marker_palette['1'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['1'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 01", showlegend = False)
        
        st.plotly_chart(fig)
        
    def display_bycustomercat_clicks_02(self):
        """
        - Displays a boxplot and a histogram of the variable 'clicks' and customer category 2.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 2]['clicks']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['2'],           marker=dict(color=self.marker_palette['2'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['2'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 02", showlegend = False)
        
        st.plotly_chart(fig)                
    
    def display_bycustomercat_clicks_03(self):
        """
        - Displays a boxplot and a histogram of the variable 'clicks' and customer category 3.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 3]['clicks']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['3'],           marker=dict(color=self.marker_palette['3'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['3'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 03", showlegend = False)
        
        st.plotly_chart(fig)       
              
    def display_bycustomercat_clicks_04(self):
        """
        - Displays a boxplot and a histogram of the variable 'clicks' and customer category 1.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 4]['clicks']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['4'],           marker=dict(color=self.marker_palette['4'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['4'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 04", showlegend = False)
        
        st.plotly_chart(fig)       
          
    
    def display_bycustomercat_clicks_05(self):
        """
        - Displays a boxplot and a histogram of the variable 'clicks' and customer category 5.
        """
        # Figure setup
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02)
        
        # Add traces
        data = self.campaigns_df[self.campaigns_df['customer_cat'] == 5]['clicks']
        
        fig.add_trace(go.Box(x=data, name=' ', boxmean=True, fillcolor=self.fillcolor_palette['5'],           marker=dict(color=self.marker_palette['5'])), row=1,col=1)
        
        fig.add_trace(go.Histogram(x=data, marker=dict(color=self.marker_palette['5'])), row=2, col=1)
        
        # Make adjustments
        fig.update_layout(height=400, width=800, title_text="Category 05", showlegend = False)
        
        st.plotly_chart(fig)       
