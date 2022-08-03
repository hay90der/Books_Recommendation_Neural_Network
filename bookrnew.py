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

















st.markdown(""" Informatics Institute for Postgraduate Studies - IIPS """)
st.markdown(""" Developed By : Hayfaa Khudir """)
