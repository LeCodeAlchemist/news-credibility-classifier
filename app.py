import streamlit as st
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer

# the stopwords
stop_words = list(pd.read_csv('data/stopwords.csv').stopwords)

# the classification models
articles_classifier = load('models/articles_classifier.joblib')
posts_classifier = load('models/posts_classifier.joblib')

# vectorization objects
articles_vect = load('models/articles_vectorizer.joblib')
posts_vect = load('models/posts_vectorizer.joblib')


# function for cleaning the new textual data
def get_clean_text(text, stopwords=set(stop_words), stemmer=PorterStemmer()):
    # remove all html tags if any
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()

    # remove all special characters and numbers. Keep letters.
    clean_text = re.sub('[^a-zA-Z]', ' ', clean_text)

    # convert the text to lowercase
    clean_text = clean_text.lower()

    # tokenize the text
    clean_text_tokens = clean_text.split()

    # stemming and removing stopwords
    filtered_words = []

    for word in clean_text_tokens:
        # removes punctuation and stop words
        if word not in stopwords and word.isalpha():
            # stems a word
            filtered_words.append(stemmer.stem(word))

    # join all the tokens
    clean_text = " ".join(filtered_words)
    return clean_text


# vectorization function
def vectorize(text, vocab):
    count_vec = CountVectorizer(vocabulary=vocab)
    text_tr = count_vec.transform([text])
    return text_tr


# Main Header
st.write("""
# News Credibility Prediction 
This app classifies if a given news article or news post from social media is credible or not.
""")

# Sidebar
st.sidebar.header('Model Options')
st.sidebar.write('Select the type of classification you want to make')


# format function
def fmt(option: str):
    return option.replace('_', ' ').capitalize()


# Model options
selected_model = st.sidebar.selectbox(label="Select A Classifier", options=["no_classifier_selected","posts_classifier", "articles_classifier"], format_func=fmt)

# Main Display

if selected_model == "no_classifier_selected":
    st.write('## Note')
    st.write("There are two options for using this application and you can pick which one you want to use")
    st.write("""
    1. If you want to classify the credibility of a post from social media applications such as twitter,
    pick the ** Posts classifier** option on the sidebar.
    """)
    st.write("""
    2. If you want to classify the credibility of a full text news article or blog, pick the ** Articles classifier **
    option on the sidebar.
    """)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if selected_model == "posts_classifier":
        st.write("## Posts Classifier")
        st.write("### Note:")
        st.write("if you are going to upload a ** .csv ** file make sure it has ** author **, ** statement ** and ** "
                 "source ** columns.")
        st.write("Your input .csv file should be able to produce at least the DataFrame shown below.")
        example_data = {
            'author': ['Name Surname'],
            'statement': ['Some statement in a social media post'],
            'source': ['the social media app where you got the post or the person who referenced the post']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)

        input_df = None

        if uploaded_file:
            feature_columns = ['author', 'statement', 'source']
            try:
                input_df = pd.read_csv(uploaded_file, usecols=feature_columns)
            except:
                st.write("The .csv file that you provided does not have the appropriate columns")
                st.write("Please upload a proper .csv file")

            st.write("### First 5 columns of the Input DataFrame")
            st.dataframe(input_df.head())

            # fill in missing values with spaces
            input_df = input_df.fillna(' ')

            # Preprocess the data
            X = input_df.author + ' ' + input_df.statement + ' ' + input_df.source
            X_clean = X.apply(get_clean_text)
            X_vect = posts_vect.transform(X_clean)
            predictions = posts_classifier.predict(X_vect)
            pred_prob = posts_classifier.predict_proba(X_vect)
            predictions = list(map(lambda x: "Yes" if x == 1 else "No", predictions))
            predictions = pd.DataFrame(predictions, columns=["credible"])
            pred_prob_df = pd.DataFrame(pred_prob, columns=["P(Not Credible)", "P(Credible)"])

            input_df = pd.concat([input_df, predictions, pred_prob_df], axis=1)

            # print out the result
            st.write("### Here are the classification results")
            st.write("The results have been appended to the end of the input dataframe to show the result for each post.")
            st.write("** Credible ** - indicates the classification results")
            st.write("** P(Not Credible) ** - indicates the probability a news post is not credible")
            st.write("** P(Credible) ** - indicates the probability a news post is credible")
            st.dataframe(input_df)

        else:
            with st.form("my_form"):
                st.write("### Input")
                # get the inputs from the user: author, statement and source
                author = st.text_input('Author')
                post = st.text_input('Post text')
                source = st.text_input('Source')

                submitted = st.form_submit_button("Classify")

            if submitted:
                st.write("## Prediction")
                input_text = author + ' ' + post + ' ' + source
                input_text_clean = get_clean_text(input_text)
                input_text_vec = posts_vect.transform([input_text_clean])
                prediction = posts_classifier.predict(input_text_vec)
                prediction_prob = posts_classifier.predict_proba(input_text_vec)
                prob_df = pd.DataFrame(prediction_prob, columns=["Probablity(Not Credible)", "Probability(Credible)"])

                if prediction[0] == 0:
                    st.write("The classification result was ", prediction[0])
                    st.write("This means the provided news post is ** Not Credible **")
                    st.dataframe(prob_df)
                else:
                    st.write("The classification result was ", prediction[0])
                    st.write("This means the provided news post is ** Credible **")
                    st.dataframe(prob_df)

    else:
        st.write("## Articles Classifier")

        st.write("### Note:")
        st.write("if you are going to upload a ** .csv ** file make sure it has ** author **, ** statement ** and ** "
                 "source ** columns.")
        st.write("Your input .csv file should be able to produce at least the DataFrame shown below.")
        example_data = {
            'title': ['The title of the news article'],
            'author': ['The person who wrote the news article'],
            'text': ['The text in the body of the news article']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)

        if uploaded_file:
            feature_columns = ['title', 'author', 'text']
            try:
                input_df = pd.read_csv(uploaded_file, usecols=feature_columns)
            except:
                st.write("The .csv file that you provided does not have the appropriate columns")
                st.write("Please upload a proper .csv file")

            st.write("### First 5 columns of the Input DataFrame")
            st.dataframe(input_df.head())

            # fill in missing values with spaces
            input_df = input_df.fillna(' ')

            # Preprocess the data
            X = input_df.title + ' ' + input_df.author + ' ' + input_df.text
            X_clean = X.apply(get_clean_text)
            X_vect = articles_vect.transform(X_clean)
            predictions = articles_classifier.predict(X_vect)
            pred_prob = articles_classifier.predict_proba(X_vect)
            predictions = list(map(lambda x: "No" if x == 1 else "Yes", predictions))
            predictions = pd.DataFrame(predictions, columns=["credible"])
            pred_prob_df = pd.DataFrame(pred_prob, columns=["P(Credible)", "P(Not Credible)"])

            input_df = pd.concat([input_df, predictions, pred_prob_df], axis=1)

            # print out the result
            st.write("### Here are the classification results")
            st.write(
                "The results have been appended to the end of the input dataframe to show the result for each post.")
            st.write("** Credible ** - indicates the classification results")
            st.write("** P(Not Credible) ** - indicates the probability a news post is not credible")
            st.write("** P(Credible) ** - indicates the probability a news post is credible")
            st.dataframe(input_df)
        else:
            with st.form("my_form"):
                st.write("### Input")
                # get the inputs from the user: author, statement and source
                title = st.text_input('Title')
                author = st.text_input('Author')
                text = st.text_input('Article text')

                submitted = st.form_submit_button("Classify")

            if submitted:
                st.write("## Prediction")
                input_text = title + ' ' + author + ' ' + text
                input_text_clean = get_clean_text(input_text)
                input_text_vec = articles_vect.transform([input_text_clean])
                prediction = articles_classifier.predict(input_text_vec)
                prediction_prob = articles_classifier.predict_proba(input_text_vec)
                prob_df = pd.DataFrame(prediction_prob, columns=["Probablity(Not Credible)", "Probability(Credible)"])

                if prediction[0] == 1:
                    st.write("The classification result was ", prediction[0])
                    st.write("This means the provided news post is ** Not Credible **")
                    st.dataframe(prob_df)
                else:
                    st.write("The classification result was ", prediction[0])
                    st.write("This means the provided news post is ** Credible **")
                    st.dataframe(prob_df)