from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(path='russian_comments_from_2ch_pikabu.csv', num_words=20000, maxlen=200):
    dataset = pd.read_csv(path, delimiter=',',  usecols=["comment", "toxic"])

    texts = dataset['comment']
    labels = dataset['toxic']

    # Разделение данных на тренировочный и тестовый наборы
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42)
    y_train = y_train.values
    y_test = y_test.values

    # Создание и обучение токенизатора
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(x_train)

    # Преобразование текста в последовательность индексов
    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_test_sequences = tokenizer.texts_to_sequences(x_test)

    # Добавление паддинга к последовательностям
    x_train_padded = pad_sequences(x_train_sequences, padding='post', maxlen=maxlen)
    x_test_padded = pad_sequences(x_test_sequences, padding='post', maxlen=maxlen)

    return (x_train_padded, y_train), (x_test_padded, y_test)
