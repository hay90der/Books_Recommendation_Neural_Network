# Import important libraries 
import numpy as np
import matplotlib.pyplot as plt
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
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

from colorama import Fore, Back, Style
def printmd(string):
    display(Markdown(string))

# Function that get book recommendations based on the cosine similarity score of books tags
def corpus_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    
    # create an Empty DataFrame object
    df = pd.DataFrame(titles.iloc[book_indices])
    df['authors'] = books.loc[books.index.values,"authors"]
    df['similarity_score'] = [row[1] for row in sim_scores]
    printmd("<h3> <b style='color:red;'>Your Recommendation books for book <i style='color:black;'>" + str(bookname) +"</i> is</b></h3><br>")
    
    return df 
    # return titles.iloc[book_indices]

# corpus_recommendations("The Hobbit")

defbook = 'The Hobbit'
bookname = st.text_input('Enter your book name',defbook)
#bookname = input()

st.table(corpus_recommendations(bookname))

# Encryption and Decryption book name


# Enter plain text of any byte (stream)
enc_bookname = str(bookname)


# A 256 bit (32 byte) key chosen
key = "abhijit#4387926131r513f124597851"

# CTR mode chosen for stream cipher
aes = pyaes.AESModeOfOperationCTR(str.encode(key))

# cipher text creation
e = aes.encrypt(enc_bookname)
# or, use this directly
# e = pyaes.AESModeOfOperationCTR(str.encode(key)).encrypt(i)

printmd("<h3> <b style='color:red;'>The Encrypted Book name is: <i style='color:black;'> " + str(e) + "</i></b></h3><br>")

#print("\n The Encrypted Book name is : \n", e)


# decrypting cipher text
# The counter mode of operation maintains state, so decryption requires a new instance be created
aes = pyaes.AESModeOfOperationCTR(str.encode(key))
d = aes.decrypt(e)
# or, use this directly
#d = pyaes.AESModeOfOperationCTR(str.encode(key)).decrypt(e)

printmd("<h3> <b style='color:red;'>The Decrypted Book name is:  <i style='color:black;'> " + str(d) +" </i></b></h3><br>")


st.markdown("""Informatics Institute for Postgraduate Studies - IIPS """)
st.markdown(""" Developed By : Hayfaa Khudir """)
