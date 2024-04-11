import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

text = "This is an example sentence for preprocessing."
processed_text = preprocess(text)

with open('processed_data.txt', 'w') as f:
    for token in processed_text:
        f.write(token + '\n')
