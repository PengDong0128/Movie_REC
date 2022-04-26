import os
import math
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from datetime import datetime
import re
import string

def merge_title_release_date(row):
    """
    create a movie title with format : Movie Name (release year)
    """
    try:
        dt = row[0]
        if '/' in dt:
            y = dt.split('/')[-1]
            if int(y)>22:
                y = '19'+y
            else:
                y = '20'+y
        elif '-' in dt:
            y = dt.split('-')[0]
        return f"{row[1]} ({y})"
    except Exception as e:
        failed_title_release_date_match.append(row)
        logging.debug(e)
        return ''

def clean_title(s):
    """
    clean title for mismatch titles.
    What this function does:
    1. lower and split
    2. remove all content in quote except release year,example:
        ABC (abc) (1991) -> ABC (1991)
    3. remove punctuations
    4. remove [the, a, an]
    5. tokenize title and sort tokens, then rejoin them
    """
    s = s.lower()
    y = s.split(' ')[-1]
    s = re.sub(r"\(.*\)",'',s)
    s = re.sub(r"[^\w\s]",'',s)
    ss = s.split()+[y]
    return ' '.join([s.strip() for s in sorted(ss) if s not in ['the','a','an']])


### import data ###
users = pd.read_csv(
    "./data/ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
)

ratings = pd.read_csv(
    "./data/ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
)

movies = pd.read_csv(
    "./data/ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"]
)

movies_meta = pd.read_csv('./data/ml-1m/movies_metadata.csv')

### import ends ###

### match movies with movies_meta ###
#create new title for movies_meta dataset
#create new title for movie_meta
failed_title_release_date_match = []
title_new = [merge_title_release_date(x) for x in zip(movies_meta['release_date'],movies_meta['title'])]
movies_meta['title'] = title_new

# for duplicate titles, take the one with more vote
movies_meta = movies_meta.sort_values(by = 'vote_count',ascending=False)
movies_meta=movies_meta.fillna({'vote_count':0})
movies_meta = movies_meta[movies_meta.title!='']
movies_meta['rk'] = movies_meta.groupby('title')['vote_count'].rank(method='first',ascending=False)
movies_meta = movies_meta[movies_meta.rk==1]

# merge movie_meta with movies
movies_meta_clean = movies_meta[movies_meta['title']!=''][['title','overview']]
movies = pd.merge(movies,movies_meta_clean,on='title',how='left')
# separate movies with successfully matched ones and unmatched ones
movies_clean = movies[~movies.overview.isnull()]
movies_mismatch = movies[movies.overview.isnull()]


# clean title and do a fuzzy match
movies_mismatch['title_clean'] = movies_mismatch['title'].apply(clean_title)
movies_meta_clean_cp  = movies_meta_clean.copy()
movies_meta_clean_cp['title_clean'] = movies_meta_clean_cp['title'].apply(clean_title)
movies_mismatch = pd.merge(movies_mismatch.drop('overview',axis=1),
                          movies_meta_clean_cp[['title_clean','overview']],
                          on='title_clean',
                          how='left')

movies_mismatch = movies_mismatch\
    .drop('title_clean',axis=1)
# add fuzzy matched ones with original ones
movies = movies_clean.append(movies_mismatch)

## check duplicate
## the fuzzy match might contain duplicates, here I manually cleaned them (just 1 pair)
# tmp = movies.groupby('movie_id',as_index=False).title.count()
## tmp.sort_values(by='title',ascending=False)
## movie_id == 704 duplicated

##movies[movies.movie_id==704]

#movies_meta_clean_cp[movies_meta_clean_cp.title.str.contains('The Quest')]

## manually drop
##movies = movies.drop(253)

### finish matching movies with movies_meta ###

### rough categorize features in movies, users and ratings ###
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
users["age_group"] = users["age_group"].apply(lambda x: f"group_{x}")
users["occupation"] = users["occupation"].apply(lambda x: f"occupation_{x}")

movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")

ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

# create genre one hot vector
genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

for genre in genres:
    movies[genre] = movies["genres"].apply(
        lambda values: int(genre in values.split("|"))
    )
genre_vec = [','.join(x) for x in movies[genres].to_numpy().astype('str')]
movies['genre_vector'] = genre_vec

# clean movies and ratings
movies = movies.dropna(subset=['movie_id','overview','title'])
movie_list = movies['movie_id'].unique()
ratings = ratings[ratings.movie_id.isin(movie_list)]

### categorization ends ###

### create user behavior sequence###
# one user per line
ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
ratings_data = pd.DataFrame(
    data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)),
        "ratings": list(ratings_group.rating.apply(list)),
        "timestamps": list(ratings_group.unix_timestamp.apply(list))
    }
)

sequence_length = 4
step_size = 2

# create movie rating sequence

def create_sequences(values, window_size, step_size):

    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

ratings_data.ratings = ratings_data.ratings.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

del ratings_data["timestamps"]

# treat every sequence separately
ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
    "movie_ids", ignore_index=True
)

# create target movie_id and target rating (label)

ratings_data_rating = ratings_data[["ratings"]].explode("ratings", ignore_index=True)

ratings_data_transformed = pd.concat([ratings_data_movies, ratings_data_rating], axis=1)
ratings_data_transformed = ratings_data_transformed.join(
    users.set_index("user_id"), on="user_id"
)

ratings_data_transformed['target_movie'] = ratings_data_transformed.movie_ids.apply(
    lambda x: x[-1])
ratings_data_transformed['target_rating'] = ratings_data_transformed.ratings.apply(
    lambda x: x[-1]
)

ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
    lambda x: ",".join(x[:-1])
)
ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
    lambda x: ",".join([str(v) for v in x[:-1]])
)

del ratings_data_transformed["zip_code"]

ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
)

### sequence generated ###

### train test split ###

random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
train_data = ratings_data_transformed[random_selection]
test_data = ratings_data_transformed[~random_selection]

train_data.to_csv("train_data.csv", index=False, sep="|", header=True)
test_data.to_csv("test_data.csv", index=False, sep="|", header=True)

###

# save movie meta data
movies.to_csv('movie_meta.csv',index=False,sep="|",header=True)