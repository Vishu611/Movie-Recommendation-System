import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from webdriver_manager.chrome import ChromeDriverManager
import json
import bs4 as bs
import urllib.request
import pickle
import requests
import imdb
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from time import sleep

import re
import os
import seaborn as sns
import sys
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from matplotlib import pyplot as plt
import io
from collections import Counter
from nltk.corpus import wordnet
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from scrapy.selector import Selector
from selenium.webdriver.common.by import By
import time
from tqdm import tqdm
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

imdb_movies=pd.DataFrame()
def webscrapping(content):
    for movie in content.select('.lister-item-content'):
        try: 
            actors=[a.text for a in movie.find('p',class_='').find_all('a')[1:]]
            data = {
                "title":movie.select('.lister-item-header')[0].get_text().strip(),
                "year":movie.select('.lister-item-year')[0].get_text().strip(),
#                 "certificate":movie.select('.certificate')[0].get_text().strip(),
                "time":movie.select('.runtime')[0].get_text().strip(),
                "genre":movie.select('.genre')[0].get_text().strip(),
                "rating":movie.select('.ratings-imdb-rating')[0].get_text().strip(),
#                 "metascore":movie.select('.ratings-metascore')[0].get_text().strip(),
                "simple_desc":movie.select('.text-muted')[2].get_text().strip(),
                "votes":movie.select('.sort-num_votes-visible')[0].get_text().strip(),
                "director":movie.find('p',class_='').find_all('a')[0].text,
                "Cast":actors

            }
            movie_list.append(data)
        except IndexError:
            continue
    print(len(movie_list),'\n')
    return movie_list


# imdb_movies.to_csv('imdb_movies_allgenres.csv')        
# print(imdb_movies.shape)

# url_dict = {}
# HEADERS ={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
# movie_list=[]
# for year in range(2000,2022):
#     threads = min(50, year)
#     for i in range (1,5000,50):
#         url = "https://www.imdb.com/search/title/?title_type=feature&start={}&release_date={}&sort=num_votes,desc&ref_=adv_nxt"
#         url = url.format(i,year)
#         resp = requests.get(url, headers=HEADERS)
#         content = BeautifulSoup(resp.content, 'lxml')
#         webscrapping(content)

#         #         with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
# #             executor.map(webscrapping,content)
            
#         print(url)


data = pd.read_csv('imdb_movies_allgenres.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
movies_titles=data['title'].str.split("\n",expand=True)
data['title'] = movies_titles[1]
data['comb']=data['director']+data['Cast']+data['genre']

data['title'].nunique()

data

movie = input("Movie_Name")

data

data['title'].nunique()

    
filename = 'nlp_model.pkl'
with open(filename, 'rb') as f:
    clf = pickle.load(f)

with open('tranform.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def create_similarity():
    # creating a count matrix
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return similarity

def rcmd(m):
    m = m.lower()
    similarity = create_similarity()
    if m not in data['title'].str.lower().unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
            
            i = data.loc[data['title'].str.lower()==m].index[0]
            lst = list(enumerate(similarity[i]))
            lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
            lst = lst[1:11] # excluding first item since it is the requested movie itself
            l = []
            
            for i in range(len(lst)):
                try:
                    a = lst[i][0]
                    l.append(data['title'][a])
                    print(l)
                except :
                    continue
            
    return l
    
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions(df):
    return list(df['title'].str.capitalize())

def home(df):
    suggestions = get_suggestions(df)
    return suggestions

def similarity(movie):
#     print("in similarity")
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return rc
    
def recommend(movie):
    # creating instance of IMDb
    ia = imdb.IMDb()
    # searching the name
    search = ia.search_movie(movie)
    # loop for printing the name and id
    for i in range(len(search)):
        if (search[i]['title'].lower() == movie.lower()):
            imdb_id = 'tt'+str(search[i].movieID)
            break

    # web scraping to get user reviews from IMDB site
    sys.path.insert(0,'/usr/local/bin/chromedriver')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')


    driver = webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)
    url = 'https://www.imdb.com/title/{}/reviews?ref_=tt_urv'.format(imdb_id)
    print(url)
    driver.get(url)
    print(driver.title)
    body = driver.find_element(By.CSS_SELECTOR, 'body')
    body.send_keys(Keys.PAGE_DOWN)
    body.send_keys(Keys.PAGE_DOWN)
    body.send_keys(Keys.PAGE_DOWN)
    review_date_list = []
    review_title_list = []
    author_list = []
    review_list = []
    review_url_list = []
    error_url_list = []
    error_msg_list = []
    reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')

    for d in tqdm(reviews):
        try:
            sel2 = Selector(text = d.get_attribute('innerHTML'))
            try:
                review = sel2.css('.text.show-more__control::text').extract_first()
            except:
                review = np.NaN
            try:
                review_date = sel2.css('.review-date::text').extract_first()
            except:
                review_date = np.NaN    
            try:
                author = sel2.css('.display-name-link a::text').extract_first()
            except:
                author = np.NaN    
            try:
                review_title = sel2.css('a.title::text').extract_first()
            except:
                review_title = np.NaN
            try:
                review_url = sel2.css('a.title::attr(href)').extract_first()
            except:
                review_url = np.NaN
            review_date_list.append(review_date)
            review_title_list.append(review_title)
            author_list.append(author)
            review_list.append(review)
            review_url_list.append(review_url)
        except Exception as e:
            error_url_list.append(url)
            error_msg_list.append(e)
    review_df = pd.DataFrame({
        'Review_Date':review_date_list,
        'Author':author_list,
        'Review_Title':review_title_list,
        'Review':review_list,
        'Review_Url':review_url
        })
    review_df
    reviews_status=[]
    reviews_list =[]
    for reviews in review_df['Review']:
        if (reviews is not None):
            reviews_list.append(reviews)
            # passing the review to our model
            movie_review_list = np.array([reviews])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}  
            movie_reviews_df = pd.DataFrame(movie_reviews.items(),columns={'Review','Outcome'})
    return (movie_reviews_df)

movie_list=similarity(movie)   

movie

f_Df=recommend(movie)
score = ((f_Df.Outcome.value_counts()[0])/len(f_Df))*100
score

df=pd.DataFrame()
for i in range(0,len(movie_list)):
    a=data[data['title']==movie_list[i]]
    df=pd.concat([a,df])

df['recomendation_score']=((df['rating']+score)/2) 

df.sort_values(by=['recomendation_score'],inplace=True)

print("According to the title given for search : " ,movie)


#we suggest the following

df[['title','rating','Cast','genre']]