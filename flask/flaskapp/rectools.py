import numpy as np
from pandas import DataFrame
from lightfm import LightFM

def cosine_similarities(vec, mat):
    """Calculate the cosine similarities between a vector and the columns of
    a matrix.
    Args:
        vec ((1,n) numpy array)
        mat ((m,n) numpy array)
    Returns:
        (m,) numpy array
    """
    sim = vec.dot(mat.T)
    matnorm = np.linalg.norm(mat, axis = 1)
    vecnorm = np.linalg.norm(vec)
    return np.squeeze(sim / matnorm / vecnorm)


def get_book_to_book_recs(bid, bid_to_idx, idx_to_bid, model):
    """Take a book ID and use a model to get the IDs of similar books.
    Args:
        bid (int): The book ID.
        bid_to_idx (dict):
            A map of the book ID to its position in model's corresponding
            matrix.
        idx_to_bid (dict): The converse of bid_to_idx.
        model (LightFM): A trained instance of the LightFM class.
    Returns:
        list: A sorted list of recommended book IDs.
    """
    item_reps = model.get_item_representations()[1]
    book_vec = item_reps[[bid_to_idx[bid]], :]
    sims = cosine_similarities(book_vec, item_reps)
    # Sort by descending similarity
    sims_idxes = np.argsort(-sims)
    return [idx_to_bid[idx] for idx in sims_idxes]

def get_metadata(bids, books_df, N = 10):
    """Get some metadata from a database for some books.
    Args:
        bids (list): The IDs of books for which metadata is requested.
        books_df (pd.DataFrame): The database containing book information.
        N (int): How many book IDs to process from the bids list.
    Returns:
        list: A list of dictionaries, each containing the metadata of a book.
    """
    books = []
    for bid in bids[:N]:
        book = books_df[books_df.book_id==bid]
        title = book.original_title.iloc[0]
        # non-string types correspond to missing entries
        if type(title) != str:
            title = book.title.iloc[0]
        search = book.isbn.iloc[0]
        # and sometimes, both isbns are missing
        if type(search) != str:
            search = book.isbn13.iloc[0]
        if type(search) != str:
            search = title.replace(' ', '+')
        url = 'https://isbndb.com/search/books/{}'.format(search)
        thumb = book.image_url.iloc[0]
        authors = book.authors.iloc[0]
        author, *_ = authors.split(',')
        books.append(dict(title = title, url = url, author = author,
                          cover = thumb, bid = bid))
    return books

