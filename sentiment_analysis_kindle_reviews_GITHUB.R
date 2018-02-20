library(tm)
library(gmodels)
library(e1071)
library(jsonlite)
library(RTextTools)

## Sentiment analysis for Kindle product review 

### Step 1. Exploring and preparing the data

## Loading data


kindle <- fromJSON("data/new_data.json", flatten=TRUE)

reviews <- kindle[, -3]
str(reviews)

table(reviews$Rating)

### Example of the data
reviews[106,2]


# Data preparation -- cleaning and standardizing text data

review_corpus <- VCorpus(VectorSource(reviews$Review))

review_corpus_clean <- tm_map(review_corpus, content_transformer(tolower))
review_corpus_clean <- tm_map(review_corpus_clean, removeNumbers)
review_corpus_clean <- tm_map(review_corpus_clean, removeWords, stopwords())
review_corpus_clean <- tm_map(review_corpus_clean, removePunctuation)
review_corpus_clean <- tm_map(review_corpus_clean, stemDocument)
review_corpus_clean <- tm_map(review_corpus_clean, stripWhitespace)

as.character(review_corpus_clean[[1]])

# Data preparation -- splitting text documents into words

review_dtm <- DocumentTermMatrix(review_corpus_clean)

# Data preparation -- creating training and test datasets

set.seed(123)

train_sample <- sample(16033, 10000)
str(train_sample)

review_dtm_train <- review_dtm[train_sample, ]
review_dtm_test <- review_dtm[- train_sample, ]

### Basic approach (3 levels: negative, neutral, positive)

rating_basic <- vector()
for (i in 1:16033){
  if (reviews$Rating[i] < 3){rating_basic[i] = "negative"}
  else if (reviews$Rating[i] == 3) {rating_basic[i] = "neutral"}
  else {rating_basic[i] = "positive"}
}

reviews$Rating <- rating_basic
reviews$Rating <- as.factor(reviews$Rating)

review_train_labels <- reviews[train_sample, ]$Rating
review_test_labels <- reviews[-train_sample, ]$Rating

prop.table(table(review_train_labels))
prop.table(table(review_test_labels))

# Data preparation -- creating indicator features for frequent words

review_freq_words <- findFreqTerms(review_dtm_train, 100)
str(review_freq_words)

review_dtm_freq_train <- review_dtm_train[ , review_freq_words]
review_dtm_freq_test <- review_dtm_test[ , review_freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, 'Yes', 'No')
}

review_train <- apply(review_dtm_freq_train, MARGIN = 2, convert_counts)
review_test <- apply(review_dtm_freq_test, MARGIN = 2, convert_counts)

### Step 2. Training a Naive Bayes model on the data

review_classifier <- naiveBayes(review_train, review_train_labels)

### Step 3. Evaluating model performance

review_test_pred <- predict(review_classifier, review_test)

CrossTable(review_test_pred, review_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

table(review_test_pred, review_test_labels)
recall_accuracy(review_test_pred, review_test_labels)


### Step 4. Improving model performance

review_classifier2 <- naiveBayes(review_train, review_train_labels, laplace = 1)
review_test_pred2 <- predict(review_classifier2, review_test)

CrossTable(review_test_pred2, review_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

recall_accuracy(review_test_pred2, review_test_labels)


### Step 2.1. Training a SVM model on the data
# build the data to specify response variable, training set, testing set.

set.seed(1)

test_sample <- 1:16033
test_sample <- test_sample[-train_sample]


container = create_container(review_dtm, reviews$Rating,
                             trainSize=train_sample, testSize=test_sample,virgin=FALSE)

# CREATE THE DOCUMENT-TERM MATRIX
review_dtm <- create_matrix(reviews$Review, language="english", removeNumbers=TRUE,
                                           stemWords=TRUE, removeSparseTerms=.998)

#train the model with SVM machine learning algorithm:
SVM <- train_model(container,"SVM")

#classify the testing set using the trained model.
SVM_CLASSIFY <- classify_model(container, SVM)

# accuracy table
test_labels <- reviews$Rating[-train_sample]
table(test_labels, SVM_CLASSIFY[,"SVM_LABEL"])

# recall accuracy
recall_accuracy(test_labels, SVM_CLASSIFY[,"SVM_LABEL"]) #0.7717553

# prepare the data for Google NL API, using python
r <- reviews[test_sample, 2]
str(r)
write.csv(r, file = "data/reviews_kindle_test_set.csv", fileEncoding="UTF-8")

# Next we applied Goole NL algorithm to obtain the sentiment scores for every review in the 
# test set. We got the resulted scores in a 'result.csv' file

## Loading data

GNL_scores <- read.csv('data/result.csv')

names(GNL_scores) <- 'scores'

scores_basic <- vector()
for (i in 1:6033){
  if (GNL_scores$scores[i] < -0.25){scores_basic[i] = "negative"}
  else if (GNL_scores$scores[i] <= 0.25) {scores_basic[i] = "neutral"}
  else {scores_basic[i] = "positive"}
}

GNL_scores$scores <- scores_basic
table(GNL_scores$scores, review_test_labels)
recall_accuracy(GNL_scores$scores, review_test_labels)
CrossTable(GNL_scores$scores, review_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))
