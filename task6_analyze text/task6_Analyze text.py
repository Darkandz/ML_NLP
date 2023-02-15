DATA_URL = "http://az.lib.ru/t/tolstoj_a_k/text_0180.shtml"
more_words_count = 100
tokens_more_count = 20

%tensorflow_version 1.x
import nltk
import warnings
import urllib.request
from tqdm import tqdm
from nltk import FreqDist
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download("stopwords", quiet=True)

from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor(language="ru")
opener = urllib.request.URLopener({})
resource = opener.open(DATA_URL)
raw_text = resource.read().decode(resource.headers.get_content_charset())

soup = BeautifulSoup(raw_text, features="html.parser")

for script in soup(["script", "style"]):
    script.extract()

cleaned_text = soup.get_text()
tokenized_sentences = [word_tokenize(sentence) for sentence in sent_tokenize(cleaned_text)]
predictions = [[pred.normal_form for pred in sent if pred.normal_form.isalpha()] for sent in tqdm(predictor.predict_sentences(sentences=tokenized_sentences), "sentences") ] # ИСПРАВИЛ НА ТОЛЬКО БУКВЫ
non_uniq_tokens = [word for sentence in predictions for word in sentence]

answers = {}
for i in non_uniq_tokens:
  if i in answers:
    answers[i] += 1
  else:
    answers[i] = 1

STOPWORDS = set(stopwords.words("russian"))

t = 1
counter = more_words_count + 1
while counter > more_words_count:
  counter = 0
  n_answers = []
  for i in answers:
    if (answers[i] > t):
      n_answers.append(i)
      counter += 1
  t += 1

counter = 0
for i in n_answers:
  if i in stopwords.words("russian"):
    counter += 1

counter1 = 0
for i in answers:
  if (answers[i] > tokens_more_count):
    counter1 += 1
print("\n")

print("Введите количество предложений")
print("Ответ:", len(predictions))

print("Введите количество токенов, состоящих только из букв")
print("Ответ:", len(non_uniq_tokens))

print(f"Какую долю среди {more_words_count} самых частотных токенов в произведении занимают слова, не входящие в стоп-лист?")
print("Ответ:", (more_words_count - counter) / more_words_count)

print(f"Сколько токенов встречается в тексте строго больше {tokens_more_count} раз?")
print("Ответ:", counter1)