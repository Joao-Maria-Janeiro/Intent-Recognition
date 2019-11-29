# Intent-Recognition

## Short Description

With this project the objective was, given an user input,recognize what the user intent was, this system is only able to recognize between 4 inputs: greeting, goodbye, price and images. We could feed it more data and make it able to recognize more intents but that was not the main focus of the project.

We were able to predict the intent quite well given our really small [dataset](https://github.com/Joao-Maria-Janeiro/Intent-Recognition/blob/master/intents.json). Since just recognizing the intent is not the funniest, we built a little chat bot around it, as this is one of the intent recognition's main applications. Our bot is able to get price and images of shoes contained [here](https://github.com/Joao-Maria-Janeiro/Intent-Recognition/blob/master/7004_1.csv), you say the brand and it will give you all the options for that brand, we could have done the product name instead of the brand but some of those names are quite big and complex so the brand was easy enough for the demonstration purpose.

## Technologies that will be used
For this project Tensorflow is going to be the main weapon of choice.
The technologies used:

* numpy
* nltk
* pandas
* json
* tensorflow
* keras

## Parameters choices

### Data preprocessing - [code](https://github.com/Joao-Maria-Janeiro/Intent-Recognition/blob/master/data_handler.py)
For this project, since our dataset is quite small, using an implementation such as word2vec or BERT would not have the best performance as those methods require a lot of data, so... We will use our trusty friend Bag Of Words. Although bag of words is not very good as it does not save any relationship between words nor does it handle words out of vocabulary well, with our really small dataset it is still our best option.

Alright so how will we build this bag of words? First of all we will lemmatize our data using nltk's WordNetLemmatizer, in case you are unfamiliar with lemmatization, it's a technique used to remove inflectional endings and return the base word of a given word (e.g: churches -> church; dogs -> dog; am, are, is -> be ). This is handy as we do not want our results to change based on how the user will conjugate the words, with this lemmatization we remove this issue. We will also make our system case insensitive converting all words to lowercase and also remove all stop words. 

After this word play is done, as you know our model only takes numbers as inputs so it is time to convert our sentences into vectors. To do so we will use sklearn's TfidfVectorizer, if you don't know what Tf-Idf is, it's bag of words but instead of saving 1 in the word positions it's saves the number of occurrences, this is important as some repeated words might have more meaning.
 This Tf-IDF vectorizer will use 1-grams (unigrams) and 2-grams (bigrams), so our vectorizer will create vectorizations for single words and for pairs of words, this is good as some pairs of words have more meaning together than apart, we could use larger n-grams but this would not give us much better accuracy.

### The model - [Code](https://github.com/Joao-Maria-Janeiro/Intent-Recognition/blob/master/main.py)
For the model we will simply use 3 densely connected layers, the input layer, the middle layer with the rectified linear unit activation function and with 2/3 input nodes + output nodes (a generally good option) nodes and finally the output layer with, you probably guessed it, softmax activation function. 

For the compiler we will use Adam as it is the most common, the loss function is softmax cross entropy as this is a multi layer output and for metrics we will use accuracy. 

## Conclusions

Our model is quite fragile to words out of vocabulary (words that it has never seen in it's dataset), this and the whole model's performance could really see a lot of benefits from an approach like word2vec or BERT as I mentioned before, but for this we would need a lot more data so, really, the thing that could really improve this project significantly would be having more data and, with that, change our tf-idf to a BERT approach.


