library(text2vec)
library(data.table)
library(magrittr)
library(readtext)
library(dplyr)
library(glmnet)
library(gmodels)
library(RTextTools)
library(jsonlite)
library(tm)


## Sentiment analysis for MovieReviews


### Step 1. Exploring and preparing the data
## Loading data

#Load IMDB reviews 
review_train = readtext(paste0(DATA_DIR, "/data/aclImdb/train_all/*"),
                        docvarsfrom = "filenames", 
                        docvarnames = c("id", "rating"),
                        dvsep = "_", 
                        encoding = "ISO-8859-1")

# Convert rating to 0-4 - negative, 7-10 - positive
review_train <- 
  review_train %>% 
  mutate(Review = text, 
         Rating = ifelse(rating < 5, 'negative', 'positive')) %>%
  select(id = doc_id, Review, Rating)


setDT(review_train)
setkey(review_train, id)
set.seed(2018L)
train_ids = review_train$id

# define preprocessing function and tokenization function
prep_fun = tolower
tok_fun = word_tokenizer

# create an iterator over tokens with the itoken() function
it_train = itoken(review_train$Review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = review_train$id, 
                  progressbar = FALSE)

# to significantly improve accuracy we'll prune the vocabulary:
## remove pre-defined stopwords, very common and very unusual terms.
# create a list of stopwords

stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
               "you", "your", "yours", "yourself", "yourselves", "he", "him",       
               "his", "himself", "she", "her", "hers", "herself")

# To improve the model even more, we'll use n-grams instead of words. 
# Here we will use up to 2-grams.
# build the vocabulary with the create_vocabulary() function.

vocab = create_vocabulary(it_train, stopwords = stop_words, ngram = c(1L, 2L))
pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)

# construct a document-term matrix
vectorizer = vocab_vectorizer(pruned_vocab)

t1 = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

# Now we have a DTM and can check its dimensions.
## the DTM has 25000 rows = number of documents
## the DTM has 86189 columns = number of unique terms.

dim(dtm_train)

# fit the model. 
# logistic regression model with an L1 penalty and 5 fold cross-validation.


NFOLDS = 5
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = review_train[['Rating']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))

plot(glmnet_classifier)
print(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

# check the model’s performance on test data

review_test = readtext(paste0(DATA_DIR, "/data/aclImdb/test_all/*"),
                       docvarsfrom = "filenames", 
                       docvarnames = c("id", "rating"),
                       dvsep = "_", 
                       encoding = "ISO-8859-1")

review_test <- 
  review_test %>% 
  mutate(Review = text, 
         Rating = ifelse(rating < 5, 'negative', 'positive')) %>%
  select(id = doc_id, Review, Rating)


setDT(review_test)
setkey(review_test, id)
set.seed(2018L)
test_ids = review_test$id

it_test = review_test$Review %>% 
  prep_fun %>% tok_fun %>% 
  # turn off progressbar because it won't look nice in rmd
  itoken(ids = review_test$id, progressbar = FALSE)


dtm_test = create_dtm(it_test, vectorizer)

preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]

preds_basic <- vector()
for (i in 1:length(preds)){
  if (preds[i] <= 0.375){preds_basic[i] = "negative"}
  else if (preds[i] < 0.625) {preds_basic[i] = "neutral"}
  else {preds_basic[i] = "positive"}
}

table(preds_basic, review_test$Rating)
CrossTable(preds_basic, review_test$Rating, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE,  dnn = c('predicted', 'actual'))
recall_accuracy(preds_basic, review_test$Rating)




# validate model on kindle reviews

kindle <- fromJSON("data/new_data.json", flatten=TRUE)

rating_kindle_basic <- vector()
for (i in 1:length(kindle$Rating)){
  if (kindle$Rating[i] <= 3){rating_kindle_basic[i] = "negative"}
  else if (kindle$Rating[i] == 3){rating_kindle_basic[i] = "neutral"}
  else {rating_kindle_basic[i] = "positive"}
}

reviews <- kindle[, -3]

review_valid <- 
  reviews %>% 
  mutate(rating = rating_kindle_basic,
         id = seq.int(nrow(reviews)))


setDT(review_valid)
setkey(review_valid, id)
set.seed(2018L)
valid_ids = review_valid$id

it_valid = review_valid$Review %>% 
  prep_fun %>% tok_fun %>% 
  # turn off progressbar because it won't look nice in rmd
  itoken(ids = review_valid$id, progressbar = FALSE)


dtm_valid = create_dtm(it_valid, vectorizer)

preds_valid = predict(glmnet_classifier, dtm_valid, type = 'response')[,1]

preds_valid_basic <- vector()
for (i in 1:length(preds_valid)){
  if (preds_valid[i] <= 0.4){preds_valid_basic[i] = "negative"}
  else if (preds_valid[i] < 0.6) {preds_valid_basic[i] = "neutral"}
  else {preds_valid_basic[i] = "positive"}
}
table(preds_valid_basic, review_valid$rating)
CrossTable(preds_valid_basic, review_valid$rating, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))
recall_accuracy(preds_valid_basic, review_valid$rating)



# validate model on TEST kindle reviews

kindle <- fromJSON("data/new_data.json", flatten=TRUE)

set.seed(123)

train_sample <- sample(16033, 10000)


rating_kindle_basic <- vector()
for (i in 1:length(kindle$Rating)){
  if (kindle$Rating[i] < 3){rating_kindle_basic[i] = "negative"}
  else if (kindle$Rating[i] == 3){rating_kindle_basic[i] = "neutral"}
  else {rating_kindle_basic[i] = "positive"}
}

reviews <- kindle[-train_sample, -3]

review_valid <- 
  reviews %>% 
  mutate(rating = rating_kindle_basic[-train_sample],
         id = seq.int(nrow(reviews)))


setDT(review_valid)
setkey(review_valid, id)
set.seed(2018L)
valid_ids = review_valid$id

it_valid = review_valid$Review %>% 
  prep_fun %>% tok_fun %>% 
  # turn off progressbar because it won't look nice in rmd
  itoken(ids = review_valid$id, progressbar = FALSE)


dtm_valid = create_dtm(it_valid, vectorizer)

preds_valid = predict(glmnet_classifier, dtm_valid, type = 'response')[,1]

preds_valid_basic <- vector()
for (i in 1:length(preds_valid)){
  if (preds_valid[i] <= 0.4){preds_valid_basic[i] = "negative"}
  else if (preds_valid[i] < 0.6) {preds_valid_basic[i] = "neutral"}
  else {preds_valid_basic[i] = "positive"}
}
table(preds_valid_basic, review_valid$rating)
CrossTable(preds_valid_basic, review_valid$rating, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))
recall_accuracy(preds_valid_basic, review_valid$rating)


