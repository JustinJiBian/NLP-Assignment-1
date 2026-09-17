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


class DANetwork(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding):
        super(DANetwork, self).__init__()
        self.embeddings = embedding_layer
        
        # 300 -> 200 -> 100 -> 2
        self.fc1 = nn.Linear(300, 200)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(200, 100)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        # x_batch shape: [B, T]
        E = self.embeddings(x_batch)
        
        # Create mask tensor M: M[i, t] = 1 if X[i, t] != 0 else 0
        M = (x_batch != 0).unsqueeze(-1).float()
        
        # Mask out PAD embeddings and average non-padded vectors
        masked_sum = torch.sum(E * M, dim=1)           # [B, 300]
        lengths = torch.sum(M, dim=1).clamp(min=1.0)  # [B, 1]
        v = masked_sum / lengths                       # [B, 300]
        
        # Feedforward pass
        h1 = self.drop1(self.relu1(self.fc1(v)))
        h2 = self.drop2(self.relu2(self.fc2(h1)))
        logits = self.fc3(h2)                          # [B, 2]
        
        return logits

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        # x_batch shape: [B, T]
        # 1. Look up embeddings: E shape [B, T, 300]
        E = self.embeddings(x_batch)
        
        # 2. Create mask tensor M: M[i, t] = 1 if X[i, t] != 0 else 0
        # M shape: [B, T, 1]
        M = (x_batch != 0).unsqueeze(-1).float()
        
        # 3. Mask out PAD embeddings and average non-padded vectors: v = sum(M * E) / sum(M)
        masked_sum = torch.sum(E * M, dim=1)           # [B, 300]
        lengths = torch.sum(M, dim=1).clamp(min=1.0)  # [B, 1] (prevent division by zero)
        v = masked_sum / lengths                       # [B, 300]
        
        # 4. Feedforward pass
        h1 = self.drop1(self.relu1(self.fc1(v)))
        h2 = self.drop2(self.relu2(self.fc2(h1)))
        logits = self.fc3(h2)                          # [B, 2]
        
        return logits
        
class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, network: nn.Module, indexer: Indexer):
        self.network = network
        self.indexer = indexer

    def predict(self, ex_words: List[str]) -> int:
        self.network.eval()
        
        # Single example inference (shape: [1, seq_len])
        indices = [self.indexer.index_of(w) if self.indexer.index_of(w) != -1 else 1 for w in ex_words]
        x_tensor = torch.tensor([indices], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.network(x_tensor)
            return torch.argmax(logits, dim=1).item()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    # Set fixed seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    indexer = word_embeddings.word_indexer
    embed_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)
    embed_layer.padding_idx = 0  # 0 corresponds to PAD token

    model = DANetwork(embed_layer)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32
    epochs = 5

    for epoch in range(epochs):
        random.shuffle(train_exs)
        model.train()
        
        for i in range(0, len(train_exs), batch_size):
            batch = train_exs[i:i + batch_size]
            B = len(batch)
            T = max(len(ex.words) for ex in batch)
            
            # Create tensor X of shape [B, T], filled with PAD index 0
            X = torch.zeros((B, T), dtype=torch.long)
            y = torch.zeros(B, dtype=torch.long)
            
            for b_idx, ex in enumerate(batch):
                # Map words to indices (1 for UNK if missing)
                word_indices = [indexer.index_of(w) if indexer.index_of(w) != -1 else 1 for w in ex.words]
                X[b_idx, :len(word_indices)] = torch.tensor(word_indices, dtype=torch.long)
                y[b_idx] = ex.label
                
            # Forward pass, loss calculation, and optimization
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    return NeuralSentimentClassifier(model, indexer)
