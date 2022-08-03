# Import important libraries 
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
import streamlit as st
import pyaes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



st.header('Book recommendation system using Neural Network')
st.markdown("""---""")

st.header('Please Enter your book name')





books = pd.read_csv("books.csv", encoding = "ISO-8859-1")


ratings = pd.read_csv("pdf2dee.csv", encoding = "ISO-8859-1")

book_tags = pd.read_csv("book_tags.csv", encoding = "ISO-8859-1")

tags = pd.read_csv("tags.csv")


tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')

to_read = pd.read_csv("to_read.csv")












tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(books['authors'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')


temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()


books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')

books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))

tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
















st.markdown(""" Informatics Institute for Postgraduate Studies - IIPS """)
st.markdown(""" Developed By : Hayfaa Khudir """)
