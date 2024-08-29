DS/ML salary prediction - project description

Objective: showcase proficiency in pyspark, mlflow and databricks through real data and ML project.

ML project objective: predict salary.

Datasets : Data scrubbed from ai-jobs.net into parquet file containing following columns:

title - job title

location - including city, country, region and others

salary_etc - information block containg salary info, seniority level, working hour , etc..

company_name - company name

company_description -short info on company

blocks - html code of scrubbed web page.

id - ID

Target: min salary offered

Metrics : Mean Absolute Error

Enviroment : project was initialy run on Databricks Community Edition platfrom. Due to platform compute limiation was eventaully moved to Google Colab. However it still can be run with no modifitcation on Databricks.

Requirements: account on Databricks needed to run MLFlow server. Can be setup on free Community Edition version.



