import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import pandas as pd
import data_handler


products = pd.read_csv("7004_1.csv", error_bad_lines=False)

training, output, labels, data = data_handler.preprocess_data()
training = np.array(training)
output = np.array(output)

model = keras.Sequential([
    keras.layers.Dense(len(output[1]), input_shape=(len(training[1]),)),
    keras.layers.Dense((2/3 * (len(training[1]))) + len(output[1]), activation='relu'),
    keras.layers.Dense(len(output[1]), activation="softmax")
])

model.compile(optimizer="adam", loss=tf.losses.softmax_cross_entropy, metrics=["accuracy"])

model.fit(training, output, epochs=1000)

tag = ""
while tag != "goodbye":
    user_input = input("User: ")
    transformed, lemmatized_user_input = data_handler.handle_predict_data(user_input)
    results = model.predict(transformed)
    results_index = np.argmax(results)
    tag = labels[results_index]
    
    temp_products = []

    if tag == "price" or tag == "images":
        for index, row in products.iterrows():
            if str(row["brand"]).lower() in str(lemmatized_user_input[0]):
                temp_products.append(row["name"])
        print("Robot: We have these available options, please select one:\n")
        for idx, product in enumerate(temp_products):
            print(str(idx) + ": " + product)
        product = products[products.name == temp_products[int(input("Choose a number: "))]]
        if tag == "price":
            print(str(product["prices.amountMin"].values[0]) + str(product["prices.currency"].values[0]) + " - " + str(product["prices.amountMax"].values[0]) + str(product["prices.currency"].values[0]))
        else:
            print("Image URL:" + str(product["imageURLs"].values[0]))
        print("You can find them here: " + str(product["prices.sourceURLs"].values[0]))
    else:
        for intent in data["intents"]:
            if intent["tag"] == tag:
                print("Robot: " + random.choice(intent['responses']))
        
