import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tflearn

import pickle as pkl
import numpy
import random
import json


number = 8
net = tflearn.input_data(shape=[None, 42])
net = tflearn.fully_connected(net, number)
net = tflearn.fully_connected(net, number)
net = tflearn.fully_connected(net, number)
net = tflearn.fully_connected(net, 7, activation='softmax')

net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load('./models/sam')


with open("data.json") as f:
  data = json.load(f)


with open("data.pkl", "rb") as f:
  words, labels, training, output = pkl.load(f)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat(mssg):
  results = model.predict([bag_of_words(mssg, words)])[0]
  results_index = numpy.argmax(results)
  tag = labels[results_index]
  for tg in data["intents"]:
    if tg["tag"] == tag:
      responses = tg["responses"]
      return random.choice(responses)
    else: 
      return "I don't understand. Try again!"


def main():
    while True:
        mssg = input("You: ")
        output = chat(mssg)
        print(output)

main()