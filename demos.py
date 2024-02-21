import pickle

# Загрузка модели
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Загрузка vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Пример новой новости
new_news = ""

# Преобразование новой новости в векторное представление
X_new = vectorizer.transform([new_news])

# Использование модели для предсказывания
prediction = model.predict(X_new)

print(f"Новость классифицирована как: {prediction[0]}")
