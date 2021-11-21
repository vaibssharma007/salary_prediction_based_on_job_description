
# Job Salary Prediction Based on Job description
### Deep learning model using NLP to predict job salary based on Indeed job postings

## Description
One of the most important things in Job Search is knowing where you are right now and where you want to go next. The mindset of a job seeker or a career switcher is very much like a product manager, trying to find the best product and market fit. Smart product managers know what their customers (Hiring Managers) want. So they carefully find out their needs which are expressed through top qualifications, "what you must have", in the job descriptions. After identifying what is commonly needed in the market they try to customize their products aka resumes, cover letters, online profiles, portfolios to demonstrate these characteristics and traits as much as possible. The purpose of this project is to use data transformation and machine learning to create a model that will predict a salary in various countries when given years of experience,education level.

How? This project will scrap 1000+ jobs in data/analytics fields in major U.S. cities and information such as job title, company, company review, city, job description, salary range will be used to generate insights about certain job category. And then combining with other factors such as employment/unemployment rate, Cost of living index, median salary by occupation, a predictive model will be built to predict the salary, and give job seeker more information on location choices.


## Model deployed as Web-Application AP at:📳 
To make this tool accessible to non-technical users, I created a flask app and deployed to heroku . Users can just copy a full job description in data related fields and paste here to get the salary range.
### Live Link : https://data-job-salary-predictor.herokuapp.com.

## Flow Chart :

![137633247-edd4a03d-a5d6-4080-8c98-352dabf1c4c2](https://user-images.githubusercontent.com/91024630/142774306-2daed2e6-a8fb-41b2-b491-e1d9eae51fa7.png)


## Dependencies:
For this project, the following tools were used:

1. Tensorflow 2 for building and training the model
2.Python 3 Pandas NumPy Seaborn Scikit-learn Matplotlib Jupyter
3. Flask for implementing the server side
4. HTML5, CSS3, JavaScript (with Web Speech API and particles.js) on the front-end.

## IO Screenshots:📷 <br>

### Enter Job Description :-

![Screenshot (22)](https://user-images.githubusercontent.com/91024630/142774239-c84d3a7e-595d-4201-b166-27f7777e0f23.png)

### Salary prediction :-

![Screenshot (23)](https://user-images.githubusercontent.com/91024630/142774244-5c74ca1f-ec00-4e2c-912c-15ded688b8c0.png)
)
![Screenshot (24)](https://user-images.githubusercontent.com/91024630/142774247-5d40b2d9-0bdf-42ea-a14f-541acbe4bbad.png)


## Dataset
- Indeed.com, 10,000+ jobs were scrapped from Indeed.com
- Cost of Living Indexes by City

## Model
### Business Understanding
This project aims to predict salary for specific fields - data related positions. Therefore, the training and test data is better to be limited within data/analytics related job positions to get more valuable features. When scrapping the data, we make sure one of these keywords exist in the whole job description: 'data analytics','data science','analysis'.

### Data Understanding
Among 10,000+ jobs scrapped from Indeed.com, only 8% of them have an explicit salary tag attached to the position. We decided to only use the 963 entries for the modeling process.

We explored the data by using word cloud to understand the common keywords appeared in the job description, and create visualization on salary and location to understand the distribution. The full data preparation and EDA notebook is here.

### Data Preparation
Several steps are envolved in the data preparation step:

Clean salary data and make sure they are all converted to a yearly rate. If salary is a range, use the average. Then create salary bins for modeling purpose.
Clean location data so that all jobs are in city/DMA level, not zipcode, or county
Initial cleaning on job description text: remove stopwords and special characters, tokenize sentences into words

### Modeling
7 models were built and tuned to get the best performing classification model. These 7 models are basically the combination of vectorization/embedding techniques + machine learning/deep learning models. You can find the detailed explanation and code in the Modeling notebook.

- Model 1: Count Vectorizer + TF-IDF Transformation + Classification Models (Random Forest/SGD/SVM)
- Model 2: Word2Vec Embedding + Classification Models (Random Forest/SGD)
- Model 3: Pre-trained GloVe Embedding + Classification Models (Random Forest/SGD)
- Model 4: Word Embeddings + Simple Neural Network Model
- Model 5: Pre-trained GloVe Embedding + Deep Neural Network Model
- Model 6: Mixed Input Network Model with City, Rate Type, and Cost of Living Indexes
- Model 7: Just use Job Title, City, Cost of Living Indexes + Classification

### Accuracy
Train/Test Accuracy is the primarily metrics to evaluate the classification models. We also take processing time, reusability of the model into consideration. Our best performing model is surprising Model 1 with the highest accuracy (0.71) and shortest processing time.

### Scope of Improvement
Job salary range can change from time to time depends on economic situation and job market. Therefore the model needs to be constantly trained with refreshed data to produce up to date results. Most of the positions were pulled in late May 2020 when COVID19 impacted North America so the salary range during this period of time may not reflect the "normal".

We scrapped the full job description from Indeed as there is no standard way to parse out company detail, skills requirements and benefits so there are noises within the job description text to extract features.

The size of the salary labeled data is small. In the future, scrape more data with salary to enhance the model.
