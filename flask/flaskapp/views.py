from flask import render_template
from flask import request
from flaskapp import app
from flaskapp.rectools import get_metadata
from flaskapp.rectools import get_book_to_book_recs
import numpy as np
import pandas as pd
import pickle
from lightfm import LightFM
import time
import random

# load the recommender (model), and maps between matrices and book IDs
with open('opt_model.p', 'rb') as f:
    model = pickle.load(f)
with open('idx_to_bid.p', 'rb') as f:
    idx_to_bid = pickle.load(f)
with open('bid_to_idx.p', 'rb') as f:
    bid_to_idx = pickle.load(f)

# "with" unnecessary here; Pandas closes the file after reading
books_df = pd.read_csv('./data/books.csv')

def random_ids():
    now = int(time.time()*100)
    random.seed(now)
    seed_book_idxs = [random.randrange(10),
                      random.randrange(10, 100),
                      random.randrange(100, 1000),
                      random.randrange(1000, len(bid_to_idx))]
    sorted_keys = list(bid_to_idx.keys())
    sorted_keys.sort()
    seed_book_ids = [sorted_keys[idx] for idx in seed_book_idxs]
    return seed_book_ids

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/otak')
def otak():
    # having a while loop provides robustness during development/debugging
    n_seed_books = 0
    while n_seed_books < 4:
        seed_book_ids = random_ids()
        seed_books = get_metadata( seed_book_ids, books_df, len(seed_book_ids))
        n_seed_books = len(seed_books)
        print('Got {} seed books'.format(n_seed_books))
    print(seed_books)
    return render_template("otak.html", seed_books = seed_books)    

@app.route('/output')
def otak_output():
    seed_book_id = int(request.args.get('bid'))
    recs = get_book_to_book_recs(seed_book_id, bid_to_idx, idx_to_bid, model)
    # the seed book is actually the first recommendation because it's the most similar
    seed_book = get_metadata(recs, books_df, 1)[0]
    rec_books = get_metadata(recs[1:], books_df, 10)[:4]
    for rec in rec_books:
        print(rec)
    return render_template("output.html", seed_book = seed_book, rec_books = rec_books)

@app.route('/aboutme')
def aboutme():
    return render_template("aboutme.html")

@app.route('/aboutotak')
def aboutotak():
    return render_template("aboutotak.html")
