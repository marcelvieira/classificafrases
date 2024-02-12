#import re
#import nltk
import pandas as pd
#import sqlalchemy
#import pyodbc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sqlalchemy import create_engine

server = 'MARCELSOUZA'
database = 'teste'
engine = create_engine('mssql+pyodbc://' + server + '/' + database + '?trusted_connection=yes&driver=SQL+Server')


df = pd.read_sql_query('select upper(TextoRequerimento) as TextoRequerimento, trim(upper(Categoria)) as Categoria FROM anulacao_Dados_2 order by Categoria', engine)

#Filtro = df['TextoRequerimento' ]
#Categoria = df['Categoria']
#print(Filtro[1]+','+Categoria[1]+',')

#print(Filtro[1])
#print(Categoria[1])

causa = df.TextoRequerimento
justificativa = df.Categoria
   # ("Estou com dor de cabeça", "saúde"),
   # ("Preciso organizar minhas finanças", "financeira"),
   # ("Estou pensando em fazer uma viagem", "pessoais"),
   # ("Como melhorar minha alimentação?", "saúde"),
   # ("Como investir meu dinheiro?", "financeira"),
   # ("Quero aprender a tocar violão", "pessoais"),
   # ("Como melhorar minha check-up?", "saúde"),
# print(sentences)

X = [causa for causa in causa]
y = [justificativa for justificativa in justificativa]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Precisão :", accuracy)

# Testando com novas frases
new_sentences = []
new_sentences_UP =[]
source = open('C:/teste_Py/texto.txt', 'w+')
source.writelines("\n"+str(accuracy))
source.writelines("\n"+str(predictions))

RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"

while new_sentences != ['Sair']: 
    new_sentences = [input("\033[;7m Digita a frase: \033[0;0m")]      
    new_sentences_UP=[(new_sentences[0].upper())]
    #print(str(new_sentences_UP) )
   # new_sentences.append(new_sentences[0].upper())
    new_sentences_vectorized = vectorizer.transform(new_sentences_UP)
    new_predictions = classifier.predict(new_sentences_vectorized)
 #  print(str(new_sentence)+" - Previsões para novas frases:", new_predictions)
    print('\033[1;36m' + str(new_sentences_UP) + ' - '+'\033[1;31m' + str(new_predictions))
    print('\033[0;0m')
    source.writelines("\n"+str(new_sentences_UP)+': '+str(new_predictions)+".")
source.close

