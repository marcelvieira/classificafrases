import bs4 as bs
import urllib.request
import re
import nltk
import heapq
import pandas as pd
import sqlalchemy
import pyodbc

from sqlalchemy import create_engine

server = 'MARCELSOUZA'
database = 'teste'
engine = create_engine('mssql+pyodbc://' + server + '/' + database + '?trusted_connection=yes&driver=SQL+Server')


df = pd.read_sql_query('select top 10 * FROM anulacao_Dados', engine)
print(df)


for row in df:
    print(row)


#nltk.download()

#source = 'Anulação de matricula por motivo de gravidez  Motivo de: Eu estou gravida e vou dar á luz em Abril e depois vou precisar de ficar 3 meses(Abril,maio,Junho)  em casa e depois tens os exames finais sem acompanhamento das aulas era impossível fazer porque vou perder muitas aulas, e em Julho volto para Luanda porque ia fazer o 1º ano lectivo e a dissertação ia escrever a distancia ou fazer o estagio em Angola.E achei melhor anular a matricula'
#source = urllib.request.urlopen("https://pt.wikipedia.org/wiki/Conselho").read()
source = open('C:/teste_Py/texto.txt', 'r')
palavras =[]
for linha in source:
    linha = linha.strip()
    palavras.append(linha)
source.close()
print(palavras[0])

soup = bs.BeautifulSoup(source.read,'lxml')

text = ''

for paragraph in soup.find_all('p'):
    text += paragraph.text
    print(text)
text = re.sub(r'[[0-9]*]',' ', text)
text = re.sub(r'.',' ', text)
text = re.sub(r's+',' ', text)
#print(text)
clean_text = text.lower()
clean_text = re.sub(r'W', ' ', clean_text)
clean_text = re.sub(r'd', ' ', clean_text)
clean_text = re.sub(r's+', ' ', clean_text)


sentences = nltk.sent_tokenize(text)

stop_words = nltk.corpus.stopwords.words('portuguese')
#temp = open('c:/teste_Py/texto.txt', 'w+')
#temp.write(clean_text)
#temp.close()

word2count = {}
for word in nltk.word_tokenize(clean_text):
   if word not in stop_words:
      if word not in word2count.keys():
         word2count[word] = 1          
      else:
         word2count[word] += 1 #conta a frequência das palavras 
         for key in word2count.keys(): 
            word2count[key] = word2count[key] /max(word2count.values()) #transforma em porcentagem
print(clean_text)
sent2score = {}

for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 30:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

best_sentences = heapq.nlargest(10, sent2score,key=sent2score.get)

print("-------------------------------------------------------------n")
for sentence in best_sentences:
    print(sentence)