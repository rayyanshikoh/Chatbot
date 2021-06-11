import tensorflow as tf
import tflearn

import numpy
import json
import pickle as pkl
import random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Loading Data
with open("data.json") as file:
    data = json.load(file)

# Lists we will be using
words = []
labels = []
docs_x = []
docs_y = []

# Preprocessing our data
for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Preparing the data for training, including bag of words
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

print(len(training[0]))
print(len(output[0]))

with open("data.pkl", "wb") as f:
        pkl.dump((words, labels, training, output), f)

# Training the Deep Neural Network (DNN)
tf.compat.v1.reset_default_graph()

number = 8

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, number)
net = tflearn.fully_connected(net, number)
net = tflearn.fully_connected(net, number)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=9,show_metric=True)
model.save("./models/sam")



# Creating a simple program to communicate with our AI
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)



def chat():
    print("Start Talking with the Bot (type quit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)

        tag = labels[results_index]

        if results[results_index] > 0.7:

            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            
            print(random.choice(responses))
        else: 
            print("I don't understand. Try again!")

chat()