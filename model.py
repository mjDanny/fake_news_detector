import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка данных
data = pd.read_csv('./data_set/fake_news.csv')

# Объединение всех признаков в один DataFrame
X = data[['Unnamed: 0', 'title', 'text']]
y = data['label']  # целевые значения

# Преобразование текстовых данных в числовые вектора с помощью TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X['text'])

# Сохранение vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Сохранение модели
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Тестирование модели
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели : {accuracy:.2f}')