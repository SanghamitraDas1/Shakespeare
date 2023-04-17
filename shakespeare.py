import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import glob, nltk, os, re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False) 

st.markdown('''
# Analyzing Shakespeare Texts
''')

# Create a dictionary (not a list)
books = {" ":" ","A Mid Summer Night's Dream":"Shakespeare/summer.txt","The Merchant of Venice":"Shakespeare/merchant.txt","Romeo and Juliet":"Shakespeare/romeo.txt"}

# Sidebar
st.sidebar.header('Word Cloud Settings')
max_word = st.sidebar.slider("Max Words",min_value=10, max_value=200, value=100, step=10)
max_font = st.sidebar.slider("Size of largest Word",min_value=50,max_value=350,value=150,step=10)
size = st.sidebar.slider("Image width",min_value=100,max_value=800,value=200,step=10)
#size = st.sidebar.slider("Image width",min_value=1,max_value=8,value=4,step=1)
random = st.sidebar.slider("Random State",min_value=20,max_value=100,value=50,step=1)
remove_stop_words = st.sidebar.checkbox("Remove Stop Words?",value=True)
st.sidebar.header('Word Count Settings')
min_word = st.sidebar.slider("Minimum Count of Words",min_value=5,max_value=100,value=50,step=10)

## Select text files
image = st.selectbox("Choose a text file", books.keys())

## Get the value
image = books.get(image)

raw_text = ""
if image != " ":
    stop_words = []
    raw_text = open(image,"r").read().lower()
    nltk_stop_words = stopwords.words('english')

    if remove_stop_words:
        stop_words = set(nltk_stop_words)
        stop_words.update(['us', 'one', 'though','will', 'said', 'now', 'well', 'man', 'may',
        'little', 'say', 'must', 'way', 'long', 'yet', 'mean',
        'put', 'seem', 'asked', 'made', 'half', 'much',
        'certainly', 'might', 'came','thou','shall','would','thy','thus','thou','thee'])
        # These are all lowercase

tokens = nltk.word_tokenize(raw_text)

tab1, tab2, tab3 = st.tabs(['Word Cloud', 'Bar Chart', 'View Text'])

def word_frequency(sentence):
        sentence = "".join(sentence)
        new_tokens = nltk.word_tokenize(sentence)
        new_tokens = [t.lower() for t in new_tokens]
        new_tokens =[t for t in new_tokens if t not in stop_words]
        new_tokens = [t for t in new_tokens if t.isalpha()]
        counted = Counter(new_tokens)
        word_freq = pd.DataFrame(counted.items(),columns=["word","frequency"]).sort_values(by="frequency",ascending=False)
        return word_freq
    
tab2_word_freq = word_frequency(raw_text)

with tab1:
    words_needed = ""
    for val in tab2_word_freq['word'].head(max_word):
        val = str(val)
        val_tokens = val.split()
        words_needed += " ".join(val_tokens)+", "
    if image != " ": 
        cloud = WordCloud(
            width=size,
            background_color="black",
            max_font_size=max_font,
            stopwords=stop_words,
            random_state=random
        ).generate(words_needed)

        fig=plt.figure(
            figsize=(8,8),
            facecolor='k',
            edgecolor='k'
        )

        fig = plt.imshow(cloud)
        fig = plt.axis('off')
        fig = plt.tight_layout(pad=0)
        fig = plt.show()
        st.pyplot(fig)
    
with tab2:
    if image != " ":
        df = tab2_word_freq[tab2_word_freq["frequency"]>=min_word]
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('frequency'),
            y=alt.Y('word',sort=alt.EncodingSortField(field="word", op='count')),
            color=alt.Color('frequency', scale=alt.Scale(scheme='redyellowgreen'))
            )
        text = bar_chart.mark_text(align="left", baseline="top").encode(
        text="frequency"
        )
        bar = bar_chart + text
        st.write(bar)

with tab3:
    if image != " ":
        raw_text = open(image,"r").read().lower()
        st.write(raw_text)
    