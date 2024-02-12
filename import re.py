import re
import nltk
import pandas as pd
import sqlalchemy
import pyodbc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sqlalchemy import create_engine

server = 'MARCELSOUZA'
database = 'teste'
engine = create_engine('mssql+pyodbc://' + server + '/' + database + '?trusted_connection=yes&driver=SQL+Server')


df = pd.read_sql_query('select * FROM anulacao_Dados order by filtro_B', engine)
#df = engine.execute('select top 10 * FROM anulacao_Dados order by filtro_B')

#Filtro = df['TextoRequerimento' ]
#Categoria = df['Categoria']
#print(Filtro[1]+','+Categoria[1]+',')

#print(Filtro[1])
#print(Categoria[1])

#for i in df:
cause = df.TextoRequerimento
justificativa = df.Categoria
    #'"'+df.filtro_B+'","'+df.Categoria+'",']
   # ("Estou com dor de cabeça", "saúde"),
   # ("Preciso organizar minhas finanças", "financeira"),
   # ("Estou pensando em fazer uma viagem", "pessoais"),
   # ("Como melhorar minha alimentação?", "saúde"),
   # ("Como investir meu dinheiro?", "financeira"),
   # ("Quero aprender a tocar violão", "pessoais"),
   # ("Como melhorar minha check-up?", "saúde"),

#print(sentences)
# Separando os dados em features e labels
X = [cause for cause in cause]
y = [justificativa for justificativa in justificativa]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Acurácia do classificador:", accuracy)

# Testando com novas frases
new_sentences = ["optei por um curso mais barato em outra instituição"]
new_sentences_vectorized = vectorizer.transform(new_sentences)
new_predictions = classifier.predict(new_sentences_vectorized)
print("Previsões para novas frases:", new_predictions)

