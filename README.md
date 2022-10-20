# Movie-Recommendation-System
A movie recommendation system, also known as a movie recommender system, is an ML-based strategy to filtering or predicting consumers' film preferences based on their previous selections and behavior .In this project we recommend the user the movies based on the search movie input by him


![image](https://user-images.githubusercontent.com/109312561/197030024-05f1451a-a854-4058-a4da-fadf89d8eead.png)


**Steps:**

1) Scrape the data from IMDB site. We scrape all the movies from 2000-2022, then save it as a csv
2) We used the multi-threading concept to fasttrack the steps
3) Calculate cosine similarity on the desired columns from data and then return a list of movies
4) Consider the reviews of the inputted movie and calulate the % of good reviews
5) Create a metric using steps 3 & 4 and then suggest movies based on that

**Sample Output**


<img width="712" alt="Screenshot 2022-10-20 at 1 41 01 PM" src="https://user-images.githubusercontent.com/109312561/197031304-c221ffd7-19fe-420e-9691-da2cbfba171b.png">
