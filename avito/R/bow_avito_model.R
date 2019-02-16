library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(Matrix)
library(stringr)
library(stringi)
library(forcats)
library(glmnet)
set.seed(0)

# assuming that train.csv is downloaded an is in ./input

Sys.setlocale(locale="ru_RU") # show russian words

#---------------------------
# will not use test set, loading only training data

cat("Reading data...\n")
tr <- read_csv("./input/train.csv")

#---------------------------
cat("Preprocessing...\n")

# here you may add other stuff than this (the commented-out ones are from the recommended kernel)

trpre <- tr %>% mutate(no_img = is.na(image) %>% as.integer(),
         no_dsc = is.na(description) %>% as.integer(),
         # no_p1 = is.na(param_1) %>% as.integer(),
         # no_p2 = is.na(param_2) %>% as.integer(),
         # no_p3 = is.na(param_3) %>% as.integer(),
         # titl_len = str_length(title),
         # desc_len = str_length(description),
         # titl_capE = str_count(title, "[A-Z]"),
         # titl_capR = str_count(title, "[А-Я]"),
         # desc_capE = str_count(description, "[A-Z]"),
         # desc_capR = str_count(description, "[А-Я]"),
         # titl_cap = str_count(title, "[A-ZА-Я]"),
         # desc_cap = str_count(description, "[A-ZА-Я]"),
         # titl_pun = str_count(title, "[[:punct:]]"),
         # desc_pun = str_count(description, "[[:punct:]]"),
         # titl_dig = str_count(title, "[[:digit:]]"),
         # desc_dig = str_count(description, "[[:digit:]]"),
         user_type = factor(user_type),
         category_name = factor(category_name) %>% as.integer(),
         parent_category_name = factor(parent_category_name) %>% as.integer(),
         region = factor(region) %>% as.integer(),
         # param_1 = factor(param_1) %>% as.integer(),
         # param_2 = factor(param_2) %>% as.integer(),
         # param_3 = factor(param_3) %>% fct_lump(prop = 0.00005) %>% as.integer(),
         city =  factor(city) %>% fct_lump(prop = 0.0003) %>% as.integer(), #lumping together uncommon factors
         user_id = factor(user_id) %>% fct_lump(prop = 0.000025) %>% as.integer(),#lumping together userids not so common
         price = log1p(price), # log(price+1)
         txt = paste(title, description, sep = " "), # treating title and description together
         mday = mday(activation_date), #day of the month
         wday = wday(activation_date)) %>%  # day of the week
  select(user_id,region, city, parent_category_name,user_type,no_img,no_dsc,txt,mday,wday,deal_probability)
  # replace_na(list(image_top_1 = -1, price = -1,
  #                 param_1 = -1, param_2 = -1, param_3 = -1,
  #                 desc_len = 0, desc_cap = 0, desc_pun = 0,
  #                 desc_dig = 0, desc_capE = 0, desc_capR = 0)) %T>%
  glimpse(trpre)

rm(tr)
gc()

#---------------------------
# how to represent the txt using bag of words (from Part 1)
cat("Parsing text...\n")

it <- trpre %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>%
  itoken()

str(it)

# then it is a

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 12500) %>%
  vocab_vectorizer()

str(vect)

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>%
  fit_transform(m_tfidf)

str(tfidf)
# tf=term frequency
#Creates TfIdf(Latent semantic analysis) model.
#The IDF is defined as follows: idf = log((# documents in the corpus) / (# documents where the term appears + 1))
# Tfidf =tf*idf

rm(it, vect, m_tfidf); gc()

#---------------------------
cat("Preparing data...\n")
# design matrix for the tfidf-part of the data

Xrest <- trpre %>%
    select(-txt,-deal_probability) %>%
    sparse.model.matrix(~ . - 1, .)

idtest=1127569:1503424
idrest=1:1127569

# go for 1e5 training samples and 1e5 validation samples - and 1e5 fake test set
# the true test is kept in vault now, and not looked at for a while!

set.seed(8701)
randtrain=sample(idrest,1e5)
randvalid=sample(setdiff(idrest,randtrain),1e5)
randtest=sample(setdiff(idrest,union(randtrain,randvalid)),1e5)
# alternatively the same user should not be split?

Xtr=Xrest[randtrain,]
Ytr=trpre[randtrain,]$deal_probability

Xval=Xrest[randvalid,]
Yval=trpre[randvalid,]$deal_probability

Xtest=Xrest[randtest,]
Ytest=trpre[randtest,]$deal_probability

tfidftr=tfidf[randtrain,]
tfidfval=tfidf[randvalid,]
tfidftest=tfidf[randtest,]

#---------------------------
cat("Training model...\n")

fit=glmnet(x=Xtr,y=Ytr) #standardize=TRUE default, not include intercept (is already included)

# since we have all these data I want to use the validation set to choose the
# optimal lambda, not the cv.glmnet  -therefore just loop over the lambdas
lambdas=fit$lambda
rmse=rep(NA,length.out=length(lambdas))
for (i in 1:length(lambdas))
{
  print(i)
  thislambda=lambdas[i]
  yhats=predict(fit,newx=Xval,type="response",s=thislambda)
  rmse[i]=sqrt(mean((Yval-yhats)^2))
}
plot(lambdas,rmse)
# OLS is the best with these predictors - so, this was a test just to check that
# things are working before going on to the tfidf


fit=glmnet(x=tfidftr,y=Ytr,standardize = FALSE) #since weighted?
lambdas=fit$lambda
rmse=rep(NA,length.out=length(lambdas))
for (i in 1:length(lambdas))
{
  print(i)
  thislambda=lambdas[i]
  yhats=predict(fit,newx=tfidfval,type="response",s=thislambda)
  rmse[i]=sqrt(mean((Yval-yhats)^2))
}
plot(lambdas,rmse)
bestlambda=lambdas[which.min(rmse)]

yhattest=predict(fit,newx=tfidftest,s=bestlambda)
testrmse=sqrt(mean((Ytest-yhattest)^2))
testrmse # not really winning anything here, but, a good start :-)
plot(Ytest,yhattest,pch=20) #ups...

# lots to check out next, transforming the Y? and adding other covs?
