# text-classification
# (MULTILABEL TOXIC COMMENT CLASSIFICATION USING MACHINE LEARNING)

This paper presents a novel application of Natural Language Processing techniques to classify unstructured text into toxic and nontoxic categories. The classification of toxic comments is a well-researched area with many techniques available. However, effectively managing multi-label categorization still requires a considerable amount of work. In this thesis, we performed a classification experiment on over 1.6 million comments from the Jigsaw toxic comment challenge data available on Kaggle. We aimed to optimize a model to identify six different categories of hate speech. Initially, we implemented a baseline model using a simple vectorization technique and logistic regression. Subsequently, we compared this model with another model. After thorough analysis, we found that a fine-tuned transformer-based model called Logistic Regression yielded the best performance, achieving an accuracy of 0.93. If we talk about the toxicity of a comment, in the current century, social media has created many job opportunities and, at the same time, it has become a unique place for people to freely express their opinions. Meanwhile, among these users, there are some groups that are taking advantage of this framework and misuse this freedom to implement their toxie mindset (i.e insulting, verb sexual harassment, threads, Obscene, etc.). There are many example where toxicity can be handled by simply detecting it. Thus, a smart use of data science is able to form a healthier

environment for virtual societies.

# TF-IDF VECTORIZER

There are several techniques available to vectorize the data such as tf-idf, Count Vectorizer WordEmbedding etc, can be used to vectorize the text data. We have used tf-idf vectorizer. Term frequency Inverse document frequency(TFIDF) is a statistical formula to convert text documents into vectors based on the relevancy of the word. It is based on the bag of words model to create a matrix containing the information about less relevant and and most relevant words in the document.

                                tf(w, d) = occurrence of w in document d / total no. of world in document d    

occurrence of w in document d total number of words in document d

idf(w, D) = In(total number of document(N) in corpus D) / number of documents containing w

                                           tfidf(w, d, D) = tf(w, d) x idf(w, D)


# Logistic Regression (Benchmark Model)

Logistic regression is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not.

                                               p = \frac{1}{1 + e^{-(ax+b)}}

By using a sigmoid function we add a threshold value(k) to classify into two labels(0,1) following as-

Y=1 if p>=k else 0

In case of multilabel classification we uses an alternative One VsRest classifier that does the same as sigmoid function, but it did this for each label.

# Naive Bayes Approach

Multinomial Narve Bayesian classification is a probabilistic approach to machine learning. It is based in the Bayes Theorem following as-
  
                                                  P(A/B) = P(BIA) P(A) P(B)

The probability of A happening knowing that B has occurred could be calculated. Event B represents the evidence and A the hypothesis to be approved. The theorem runs on the assumption that all predictors/features are independent and the presence of one would not affect the other. This is the Bayes Naive simplification. The probability of one event, B, is independent from another B event occurring. The approach to classify an Internet comment as offensive or toxic would begin by studying our collection of training data labeled as toxic and non-toxic. But using OneVsRest classifier, it can be used for multilabel classification. The Multinomial Naive Bayes is used in cases where the dataset is imbalance.
