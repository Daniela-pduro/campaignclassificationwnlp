![Logo](images/readme_image_blue.png)

## Email Marketing Campaigns Optimization Model with NLP

Our project draws on data from thousands of email marketing campaigns to create a tool which predicts the outcome of a marketing message in terms of open rate, CTR (Click Through Rate) and CTOR (Click Through Open Rate) scores.

The model wants to prevent marketers from sending campaigns with predicted scores lower than their industry benchmarks without having revised the content of their messages first.

The project also provides a multiclass classifier which aims to predict the vertical or sector of a company based on the text content of its campaigns.


### App

Monitoring platform for email marketing campaigns with three main tools: Performance Analysis Tool, Vertical Detector Tool & KPIs Prediction Tool.

- **Performance Analysis Tool**: tool to analyze email marketing campaigns performance.

[Demo](https://user-images.githubusercontent.com/74190803/127731614-c7f59cd6-fa0f-4811-bed1-4752ae52c9ca.mp4)


- **Vertical Detector Tool**: tool to predict the industry or sector of a company.

[Demo-2](https://user-images.githubusercontent.com/74190803/127731673-55ecb820-c220-4960-b43c-520fc8b86e44.mp4)


- **KPIs Prediction Tool**: tool to predict Open Rate, CTR and CTOR results of a campaign. In case the campaign does not obtain the desired predicted result, it makes some recommendations based on the most succesful keywords from its industry-specific curated corpus.

[Demo-3](https://user-images.githubusercontent.com/74190803/127731697-cecea955-caf1-4bc9-b30b-f453180a7b5c.mp4)


### Table of contents

- [App](#app)
- [Installation](#installation)
- [Notebooks](#notebooks)


### Installation

[(Back to top)](#table-of-contents)

➡  **Please follow these steps to install the project's packages and dependencies:**

1. Install Anaconda.

2. Create a folder and name it "project".

3. Clone this repository inside of "project": https://github.com/Daniela-pduro/emailmarketingwnlp

4. Create an environment: `conda create --name tfm-env python=3.8`

5. Activate the environment: `conda activate tfm-env`

6. Install our project's dependencies:

`pip install jupyter==1.0.0 pandas==1.2.3 scipy==1.6.1 scikit-learn==0.24.1 langdetect==1.0.8 nltk==3.5 matplotlib==3.3.4 seaborn==0.11.1 plotly==4.14.3 gensim==4.0.0 streamlit==0.74.1 keras==2.4.3 tensorflow==2.4.1 transformers==4.4.2 spacy==3.1.0 imblearn==0.0`

7. Install Spacy's trained pipeline "es_core_news_sm": 
    
`python -m spacy download es_core_news_sm`

8. Download the zip file named "data.zip" from Google Drive.

9. Unzip this file in "project" folder. Depending on the unzip program used, an additional folder can be created. 
If that's the case, copy the folders "corpora", "datasets" and "keywords" so they are at the same level than "emailmarketingwnlp". 

10. Go to the folder "emailmarketingwnlp".

➡  **Once installed, you can run the scripts of the Main Model following these steps...**

1. Launch Jupyter Notebook from your console: `jupyter-notebook`

2. Go to `notebooks > 01_Main_Model > 10_Transforming_and_making_predictions_on_new_data`

3. Run any cell.

➡  **... or you can try our Streamlit App following these ones:**

1. Launch Jupyter Notebook from your console:`jupyter-notebook`

2. Go to `notebooks > 02_App > 01_streamlit_app`

3. Run both cells.

**Important note**: running Streamlit for the first time requires the user to introduce an email. As the prompt is not interactive, we recommend to run `streamlit hello` from the terminal (having the virtual environment activated) so that Jupyter's window does not freeze.

### Notebooks

Please, open the folder named `notebooks` to find out the different parts of this project:

#### `01_Main_Model`

It contains the methodology of the project with printed outputs. It starts with an exploratory analysis of the input data, looks into different feature engineering steps (including texts feature engineering using NLP), and finishes with the implementation of three machine learning models to detect Open Rate, CTR and CTOR results.

Notebooks `07_First_model_Model_Selection`  & `08_First_Model_predicting_sectors` within `Main_Model` contain a secondary model also called `Vertical_Model` which detects the sector or vertical of a given company based on the text content of its campaigns.

The folder named `languages` in `01_Main_Model` shows some other examples of running this `Vertical_Model` with data in various languages: Spanish, English, French and Portuguese.

#### `02_App`

It contains the scripts of our Front-End App in Streamlit.

#### `03_Scripts`

It contains nine PY files with the scripts of the most important steps developed in `01_Main_Model`.

#### `campaigns`

This module contains the classes used for the data analysis and transformations needed in `01_Main_Model`.

#### `streamlitapp`

This module contains the classes used for the data analysis and transformations needed in `02_App`.


![Footer](images/footer.png)