from django.shortcuts import render
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Create your views here.
def index(request):
    file_path = os.path.join(os.path.dirname(__file__), "fake_or_real_news.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    data = pd.read_csv(file_path)
    data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    x, y = data['text'], data['fake']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    clf = LinearSVC()
    clf.fit(x_train_vectorized, y_train)
    text = ""
    if request.method == "POST":
        text = request.POST.get('newsInput', "").strip()
    vectorized_text = vectorizer.transform([text])
    result = clf.predict(vectorized_text)
    if result == 1:
        result = "FAKE"
    else:
        result = "REAL"
    return render(request,'index.html', {"result": result})