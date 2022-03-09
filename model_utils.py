from pathlib import Path
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score, recall_score, precision_score


def __most_frequent_words(X_text, K):
    
    corpus_word_list = ' '.join(X_text).split(' ')

    from collections import Counter
    word_count = Counter(corpus_word_list)
    print("Vocabulary Size:", len(word_count))

    # Let's take 'K' most frequent words
    return word_count.most_common(K)[-10:]
    
    
def save(model, file_name):
    
    file_path = Path(file_name)

    with open(str(file_path), 'wb') as model_file:
        save_status = pickle.dump(model, model_file)
    
    return save_status


def load(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise Exception('FileNotFound', f'Model file {str(file_path)} does not exist!')

    with open(str(file_path), 'rb') as model_file:
        model = pickle.load(model_file)
        
    return model


def print_metrics(y_true, y_pred, data="test"):
    print(f"****Statistics on {data.upper()} Data****")
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print("Positive predictions: ", sum(y_pred), "/", sum(y_true))
    print(f"F1, Recall, Precision: ({f1:.2f}, {recall:.2f}, {precision:.2f})")
    return f1, recall, precision
    

def create_text_classifier():
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(class_weight='balanced', random_state=1))
                    ])


def evaluate(model, X_train, y_train, X_test, y_test):

    train_pred = model.predict(X_train)
    print_metrics(y_train, train_pred, data='train')

    y_pred = model.predict(X_test)
    
    return print_metrics(y_test, y_pred, data='test')
