import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
import re
import pandas as pd


# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')


tfidf = pickle.load(open('vec.pkl','rb'))
model = pickle.load(open('mod.pkl','rb'))

#removing punctuation
def remove_punctuation(text):
    
    return re.sub(r'[^\w\s]', '', text)

#removing stopwords
stop = set(stopwords.words('english'))
def remove_stopwords(text):

    words = [ w for w in text.split() if not w in stop]
    return ' '.join(words)

#stemmong the sentence
stemmer = nltk.stem.SnowballStemmer('english')
def stem_text(text):
    if isinstance(text, str):
         words = text.split()
         stem_words = [stemmer.stem(word) for word in words]
         return ' '.join(stem_words)
    return text

labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.title("Toxic Comment Classifier")

comment = st.text_area("Enter the comment:")
#clean the comment
clean=stem_text(remove_stopwords(remove_punctuation(comment)))


if st.button("Classify"):
    if comment:
        
        comment_vec = tfidf.transform([clean])

        # Predict using the loaded classifier
        prediction = model.predict(comment_vec)[0]
        # Display the results
        df=pd.DataFrame({"Labels": labels, "Toxicity Reasult": prediction})
        st.subheader("Predictions:")
        st.table(df)
    else:
        st.warning("Please enter a comment to classify.")

# Run it: streamlit run app.py
