import sys
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from joblib import dump
nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Load data from db and create features and targets.
    :param database_filepath: DB-filepath to read data from
    :return: Features, Targets, Target names
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    query = "SELECT * FROM disaster_messages_categories"
    # engine.table_names()
    df = pd.read_sql(query, engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, y, y.columns


def tokenize(text):
    """
    Apply various text processing to text.
    :param text: Text to process
    :return: Processed Text
    """
    # lower case
    text = text.lower()
    # remove punctuation (keep only A-Z and 0-9)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    tokens = word_tokenize(text)
    # stop words removal
    text = [word for word in tokens if word not in stopwords.words("english")]
    # Stemming
    stemmed_text = [PorterStemmer().stem(word) for word in text]
    return stemmed_text


def build_model():
    """
    Build ML-Model with MultiOutputClassifier and RandomForestClassifier. Apply TF-IDF and perform gridsearch to retrieve the best model.
    :return: ML-Model ready to be trained.
    """
    model_rf = RandomForestClassifier(class_weight='balanced')
    model = MultiOutputClassifier(estimator=model_rf)
    pipeline = Pipeline([
        # ('clean_text', CleanText()), # This is done separately before the model build to reduce time for tuning
        ('list_conversion', ConvertListToString()),
        ('tfidf', TfidfVectorizer()),
        ('cls', model),
    ])

    parameters = {
        'tfidf__max_df': [1.0],
        'cls__estimator__n_estimators': [100,],
        'cls__estimator__max_depth': [None, ],
        'cls__estimator__min_samples_split': [2,],
    }
    cv = 2

    grid_model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=cv)
    return grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model: Pre-built ML-Model
    :param X_test: Test-Features
    :param Y_test: Test-Targets (Multi-Output)
    :param category_names:
    """
    y_test_pred = model.predict(X_test)
    for index, col in enumerate(Y_test):
        print(f'Classification Report for {col}:')
        print(classification_report(Y_test.iloc[:, index], pd.DataFrame(y_test_pred).iloc[:, index]))
        print('_____________________')


def save_model(model, model_filepath):
    """

    :param model:
    :param model_filepath:
    """
    dump(model, model_filepath)


class ConvertListToString(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        text = X.apply(lambda x: " ".join(x))
        return text


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # Prepare Text data
        X = X.apply(tokenize)

        # Train Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()