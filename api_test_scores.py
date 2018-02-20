
# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

import csv

# Instantiates a client
client = language.LanguageServiceClient()

reviews = []
scores = []

with open('reviews_kindle_test_set.csv', 'r') as csvfile:
    file = csv.reader(csvfile)
    reviews = list(file)
    for i in range(len(reviews)):
        document = types.Document(content=u''.join(reviews[i]),
                                  type=enums.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        scores.append(sentiment.score)


with open('result.csv', 'w') as file_handler:
    for item in scores:
        file_handler.write("{}\n".format(item))
