import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 
import json
from sklearn.feature_extraction.text import CountVectorizer
import random
import pandas as pd

lemmatizer = WordNetLemmatizer() 
cv = CountVectorizer()


with open("intents.json") as file:
    data = json.load(file)

products = pd.read_csv("7004_1.csv", error_bad_lines=False)

labels = []
patterns_text = []
patterns_label = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        lemmatized = [ lemmatizer.lemmatize(w.lower()) for w in  nltk.word_tokenize(pattern) if w != "?"]
        patterns_text.append(' '.join(lemmatized))
        patterns_label.append(intent["tag"])
    
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
    

training = cv.fit_transform(patterns_text).toarray()

labels = sorted(labels)


output = []
out_empty = [0 for _ in range(len(labels))]

for x, patter in enumerate(patterns_text):
    output_row = out_empty[:]
    output_row[labels.index(patterns_label[x])] = 1

    output.append(output_row)

training = np.array(training)
output = np.array(output)

model = keras.Sequential([
    keras.layers.Dense(len(output[1]), input_shape=(len(training[1]),)),
    keras.layers.Dense(146, activation='relu'),
    keras.layers.Dense(len(output[1]), activation="softmax")
])

model.compile(optimizer="adam", loss=tf.losses.softmax_cross_entropy, metrics=["accuracy"])

model.fit(training, output, epochs=1000)

tag = ""
while tag != "goodbye":
    user_input = input("User: ")
    lemmatized_user_input = [' '.join([ lemmatizer.lemmatize(w.lower()) for w in  nltk.word_tokenize(user_input)])]

    results = model.predict(cv.transform(
        lemmatized_user_input
    ).toarray())
    results_index = np.argmax(results)
    tag = labels[results_index]

    # print("Robot: " + tag)

    temp_products = []

    if tag == "price":
        for index, row in products.iterrows():
            if str(row["brand"]).lower() in str(lemmatized_user_input[0]):
                temp_products.append(row["name"])
        print("Robot: We have these available options, please select one:\n")
        for idx, product in enumerate(temp_products):
            print(str(idx) + ": " + product)
        product = products[products.name == temp_products[int(input("Choose a number: "))]]
        print(str(product["prices.amountMin"].values[0]) + str(product["prices.currency"].values[0]) + " - " + str(product["prices.amountMax"].values[0]) + str(product["prices.currency"].values[0]))
        print("You can find them here: " + str(product["prices.sourceURLs"].values[0]))

            

# "patterns": ["Show me the price of the green stan smith", "Price of air jordans", "Get pictures of the latest nike kicks", "How much do pharrell hu cost", "how much is a playstation 4", "get me a cheap coat", "Where can I find calvin klein pants", "I'm missing an iphone", "Where are android phones available", "Can you get me some candles"],


    # for intent in data["intents"]:
    #     if intent["tag"] == tag:
    #         if tag == "shop":
    #             if ("image" or "picture") in  lemmatized_user_input[0]:
    #             #    print("Robot: What images do you want to see?")
    #             #    new_input = input("User: ")
    #             else: 
    #                 print("Robot: Nothing")
    #         else:
    #             print("Robot: " + random.choice(intent['responses']))


