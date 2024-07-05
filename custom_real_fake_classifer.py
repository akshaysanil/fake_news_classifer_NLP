
import re
import numpy as np
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

obj_lemm = WordNetLemmatizer()


def preprocess_and_predict(text, vectorizer, model):
    # Preprocess the text
    review = re.sub('[^a-zA-Z]', " ", text)
    review = review.lower()
    review = review.split()
    review = [obj_lemm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    print(review)
    
    # Vectorize the text using the same vectorizer fitted on the training data / load the vectorizer
    vectorizer = joblib.load(vectorizer)
    text_vector = vectorizer.transform([review]).toarray()
    
    # Ensure the text_vector has the same number of features
    required_features = vectorizer.max_features
    current_features = text_vector.shape[1]
    
    if current_features < required_features:
        # Pad the vector with zeros to reach the required number of features
        text_vector = np.pad(text_vector, ((0, 0), (0, required_features - current_features)), 'constant')
    
    # Predict using the model
    # Load model
    model = joblib.load(pa_classifier_locally)
    prediction = model.predict(text_vector)
    print(prediction)
    if prediction == 1:
        return 'Real news'
    else:
        return 'Fake news'
    # return prediction

# you can add the custom text here
custom_text = 'On August 20, 2022, a TikTok video was posted, claiming that Disney World was going to lower the drinking age to 18.\
      It was stated that Disney World was battling the Florida government in court to get a resort exemption, \
        which would allow anyone 18 and older to drink on property. The TikTok video acquired millions of views in just a couple days. \
            This story was also posted on facebook, \
      instagram, and Twitter. Shortly after, the story made it on ABC 10'
pa_classifier_locally = 'models/tf_idf_passiveAgressiveFakeNewsClassifier.pkl'
# count_vectorizer = 'models/countvectorizer_tf_idf_model.pkl'
count_vectorizer = 'models/countvectorizer_tf_idf_model.pkl'
prediction = preprocess_and_predict(custom_text, count_vectorizer, pa_classifier_locally)
print(f'Prediction for the custom text is : {prediction}')
