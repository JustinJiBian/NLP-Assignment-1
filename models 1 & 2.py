# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
from nltk.corpus import stopwords

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words("english"))

    def get_indexer(self):
        return self.indexer

    def extract_features(
        self, sentence: List[str], add_to_indexer: bool = False
    ) -> Counter:
        features = Counter()

        for word in sentence:
            word = word.lower()

            if word in self.stop_words:
                continue

            idx = self.indexer.index_of(word)

            if idx != -1:
                features[idx] += 1

        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):

    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        x = self.feat_extractor.extract_features(ex_words)

        score = sum(
            self.weights[i] * count
            for i, count in x.items()
        )

        return 1 if score >= 0 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    min_count = 2  # Words must appear at least this many times
    stop_words = set(stopwords.words('english'))
    word_counts = Counter()
    
    # 1. Build the vocabulary
    for ex in train_exs:
        for word in ex.words:
            word = word.lower()
            if word not in stop_words:
                word_counts[word] += 1
                
    indexer = feat_extractor.get_indexer()
    for word, count in word_counts.items():
        if count >= min_count:
            # Assuming add_and_get_index is the method in utils.py's Indexer
            indexer.add_and_get_index(word) 

    # 2. Train the model
    epochs = 10
    learning_rate = 0.01
    weights = np.zeros(len(indexer))
    
    for epoch in range(epochs):
        random.shuffle(train_exs)
        
        for ex in train_exs:
            x = feat_extractor.extract_features(ex.words)
            
            # Calculate score and apply sigmoid
            score = sum(weights[i] * count for i, count in x.items())
            score = np.clip(score, -500, 500) # Prevent overflow in exp
            p = 1.0 / (1.0 + np.exp(-score))
            
            error = ex.label - p
            
            # SGD gradient update
            for i, count in x.items():
                weights[i] += learning_rate * error * count
                
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_logistic_regression_schedule(
    train_exs, dev_exs, feat_extractor, schedule="constant"
):
    min_count = 2
    stop_words = set(stopwords.words("english"))

    # Build vocabulary
    word_counts = Counter()
    for ex in train_exs:
        for word in ex.words:
            word = word.lower()
            if word not in stop_words:
                word_counts[word] += 1

    indexer = feat_extractor.get_indexer()
    for word, count in word_counts.items():
        if count >= min_count:
            indexer.add_and_get_index(word)

    epochs = 10
    eta0 = 0.01
    gamma = 0.9
    tau = 1000

    weights = np.zeros(len(indexer))
    examples = train_exs.copy()

    history = {
        "iteration": [],
        "train_acc": [],
        "dev_acc": [],
        "log_likelihood": []
    }

    t = 0

    for epoch in range(epochs):
        random.shuffle(examples)

        for ex in examples:
            t += 1

            if schedule == "constant":
                eta = eta0
            elif schedule == "decay":
                eta = eta0 * (gamma ** epoch)
            elif schedule == "1/t":
                eta = eta0 / (1 + t / tau)

            x = feat_extractor.extract_features(ex.words)

            score = sum(weights[i] * count for i, count in x.items())
            score = np.clip(score, -500, 500)
            p = 1.0 / (1.0 + np.exp(-score))

            error = ex.label - p
            for i, count in x.items():
                weights[i] += eta * error * count

        # ----- evaluation after each epoch -----
        def get_accuracy(data):
            correct = 0
            for ex in data:
                x = feat_extractor.extract_features(ex.words)
                score = sum(weights[i] * count for i, count in x.items())
                pred = 1 if score >= 0 else 0
                correct += (pred == ex.label)
            return correct / len(data)

        ll = 0.0
        for ex in train_exs:
            x = feat_extractor.extract_features(ex.words)
            score = sum(weights[i] * count for i, count in x.items())
            score = np.clip(score, -500, 500)
            p = 1.0 / (1.0 + np.exp(-score))
            ll += (
                ex.label * np.log(p + 1e-12)
                + (1 - ex.label) * np.log(1 - p + 1e-12)
            )

        history["iteration"].append(t)
        history["train_acc"].append(get_accuracy(train_exs))
        history["dev_acc"].append(get_accuracy(dev_exs))
        history["log_likelihood"].append(ll)

    return LogisticRegressionClassifier(weights, feat_extractor), history


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError
