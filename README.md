# Similarity-preserving-representation-learning-for-weekly-electricity-load-forecasting
In this code repo we present a new representation learning method for time series forecasting, specifically applied to electricity load forecasting. This new representation learning method is called similarity preserving, based on similarity metric learning like contrastive learning but much more efficient and with other conceptual differences.  


The used datasets "electricity_cleaned" and "weather" are opensource, and can be easily downloaded from the The Building Data Genome Project 2 (BDG2) dataset (https://github.com/buds-lab/building-data-genome-project-2/tree/master/data). You need to download the datasets:  

*electricity_cleaned.csv: located in building-data-genome-project-2/data/meters  

*weather.csv: located in data/weather  

*metadata.csv located in metadata (optional, if you would like to test the source selection mechanism).  


*To run the code, you need to access main_zeroshot.py, put the right data directories in lines 58 and 60, install the packages listed in the file "requirements".  

*For reproducibility of the results, use the following source-target pairs (lines 64-72):  

*RL1: source= Robin_lodging_Dorthy, target=Robin_lodging_Celia  

*RL2: source=Robin_lodging_Celia, target=Robin_lodging_Dorthy  

*RL3: source=Robin_lodging_Oliva, target=Robin_lodging_Elmer  

*RL4: source=Robin_lodging_Dorthy, target=Robin_lodging_Oliva  

*HO1: source=Hog_office_Miriam, target=Hog_office_Bessie  

*HO2: source=Hog_office_Mary, target=Hog_office_Valda  

*HO3: source=Hog_office_Bessie, target=Hog_office_Bill  

*HO4: source=Hog_office_Mary, target=Hog_office_Miriam  

*HO5: source=Hog_office_Bessie, target=Hog_office_Mary  

*HO6: source=Hog_office_Miriam, target=Hog_office_Sherrie  

*HI1: source=Hog_industrial_Jeremy, target=Hog_industrial_Joanne  



