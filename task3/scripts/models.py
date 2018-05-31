# Models for word alignment
import numpy as np
from collections import defaultdict

class TranslationModel1:
    "Models conditional distribution over trg words given a src word, i.e. t(f|e)."

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = defaultdict(lambda: defaultdict(float))
        self._trg_given_src_probs = defaultdict(lambda: defaultdict(float))
                

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token, i.e. t(f|e)."
        if src_token not in self._trg_given_src_probs:
            return 1.0
        if trg_token not in self._trg_given_src_probs[src_token]:
            return 1.0
        return self._trg_given_src_probs[src_token][trg_token]

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from matrix: matrix[j][i] = p(a_j=i|e, f)"
        assert len(posterior_matrix) == len(trg_tokens)
        for posterior in posterior_matrix:
            assert len(posterior) == len(src_tokens)
        # Hint - You just need to count how often each src and trg token are aligned
        # but since we don't have labeled data you'll use the posterior_matrix[j][i]
        # as the 'fractional' count for src_tokens[i] and trg_tokens[j].
        for i, src_token in enumerate(src_tokens):
            for j, trg_token in enumerate(trg_tokens):
                self._src_trg_counts[src_token][trg_token] += posterior_matrix[j][i]
        
    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        # Hint - Just normalize the self._src_and_trg_counts so that the conditional
        # distributions self._trg_given_src_probs are correctly normalized to give t(f|e).
        trg_src_sum = defaultdict(float)
        for src_token in self._src_trg_counts:
            for trg_token in self._src_trg_counts[src_token]:
                trg_src_sum[src_token] += self._src_trg_counts[src_token][trg_token]
        
        for src_token in self._src_trg_counts:
            for trg_token in self._src_trg_counts[src_token]:
                trg_src_prob = self._src_trg_counts[src_token][trg_token]
                trg_src_prob /= trg_src_sum[src_token]
                self._trg_given_src_probs[src_token][trg_token] = trg_src_prob  
        
        self._src_trg_counts = defaultdict(lambda: defaultdict(float))

class TranslationModel2:
    "Models conditional distribution over trg words given a src word, i.e. t(f|e)."

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = defaultdict(lambda: defaultdict(float))
        self._trg_given_src_probs = defaultdict(lambda: defaultdict(float))
                

    def get_conditional_prob(self, src_token, trg_token, src_tag, trg_tag):
        "Return the conditional probability of trg_token given src_token, i.e. t(f|e)."
        if (src_token, src_tag) not in self._trg_given_src_probs:
            return 1.0
        if (trg_token, trg_tag) not in self._trg_given_src_probs[(src_token, src_tag)]:
            return 1.0
        return self._trg_given_src_probs[(src_token, src_tag)][(trg_token, trg_tag)]

    def collect_statistics(self, src_tokens, trg_tokens, src_tags, trg_tags, 
                           posterior_matrix):
        "Accumulate counts of translations from matrix: matrix[j][i] = p(a_j=i|e, f)"
        assert len(posterior_matrix) == len(trg_tokens)
        for posterior in posterior_matrix:
            assert len(posterior) == len(src_tokens)
        # Hint - You just need to count how often each src and trg token are aligned
        # but since we don't have labeled data you'll use the posterior_matrix[j][i]
        # as the 'fractional' count for src_tokens[i] and trg_tokens[j].
        for i, src_token in enumerate(src_tokens):
            for j, trg_token in enumerate(trg_tokens):
                self._src_trg_counts[(src_token, src_tags[i])][(trg_token, trg_tags[j])] += posterior_matrix[j][i]
                        
    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        # Hint - Just normalize the self._src_and_trg_counts so that the conditional
        # distributions self._trg_given_src_probs are correctly normalized to give t(f|e).
        trg_src_sum = defaultdict(float)
        for (src_token, src_tag) in self._src_trg_counts:
            for (trg_token, trg_tag) in self._src_trg_counts[(src_token, src_tag)]:
                trg_src_sum[(src_token, src_tag)] += self._src_trg_counts[(src_token, src_tag)][(trg_token, trg_tag)]
        
        for (src_token, src_tag) in self._src_trg_counts:
            for (trg_token, trg_tag) in self._src_trg_counts[(src_token, src_tag)]:
                trg_src_prob = self._src_trg_counts[(src_token, src_tag)][(trg_token, trg_tag)]
                trg_src_prob /= trg_src_sum[(src_token, src_tag)]
                self._trg_given_src_probs[(src_token, src_tag)][(trg_token, trg_tag)] = trg_src_prob  
        
        self._src_trg_counts = defaultdict(lambda: defaultdict(float))

class PriorModel1:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = defaultdict(lambda: defaultdict(float))
        self._distance_probs = defaultdict(lambda: defaultdict(float))

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        #prob = 1./(abs(1.*src_index/src_length - 1.*trg_index/trg_length) + 1)
        prob = 1.0/ src_length
        return prob
        
    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        pass
        
    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass
        
class PriorModel2:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = defaultdict(lambda: defaultdict(float))
        self._distance_probs = defaultdict(lambda: defaultdict(float))

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        prob = 1./(abs(1.*src_index/src_length - 1.*trg_index/trg_length) + 1)
        return prob

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        pass
        
    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass

        
class PriorModel3:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = defaultdict(lambda: defaultdict(float))
        self._distance_probs = defaultdict(lambda: defaultdict(float))

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        if len(self._distance_probs[src_index]) == 0: #beginning
            return 1.0 / src_length
        return self._distance_probs[src_index][(trg_index, src_length, trg_length)]

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        for src_index in range(src_length):
            for trg_index in range(trg_length):
                self._distance_counts[src_index][(trg_index,src_length,trg_length)] += posterior_matrix[trg_index][src_index]
        
    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        
        trg_src_sum = defaultdict(float)
        for src_index in self._distance_counts:
            for key in self._distance_counts[src_index]:
                trg_src_sum[key] += self._distance_counts[src_index][key]
        
        for src_index in self._distance_counts:
            for key in self._distance_counts[src_index]:
                prior_prob = self._distance_counts[src_index][key]
                prior_prob /= trg_src_sum[key]
                self._distance_probs[src_index][key] = prior_prob
        
        self._distance_counts = defaultdict(lambda: defaultdict(float))
        
class PriorModel4:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus_tags, trg_corpus_tags):
        "Add counters and parameters here for more sophisticated models."
        self._probs = defaultdict(lambda: defaultdict(float))
        for src_tags, trg_tags in zip(src_corpus_tags, trg_corpus_tags):
            for src_tag in src_tags:
                for trg_tag in trg_tags:
                    self._probs[src_tag][trg_tag] += 1.
        sum_trg = defaultdict(float)
        for src_tag in self._probs:
            for trg_tag in self._probs[src_tag]:
                sum_trg[src_tag] += self._probs[src_tag][trg_tag]
            for trg_tag in self._probs[src_tag]:
                self._probs[src_tag][trg_tag] /= sum_trg[src_tag]
        
    def get_prior_prob(self, src_tag, trg_tag, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        prob = self._probs[src_tag][trg_tag]
        return prob

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        pass
        
    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass
    
class PriorModel5:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus_tags, trg_corpus_tags):
        "Add counters and parameters here for more sophisticated models."
        self._probs = defaultdict(lambda: defaultdict(float))
        for src_tags, trg_tags in zip(src_corpus_tags, trg_corpus_tags):
            for src_tag in src_tags:
                for trg_tag in trg_tags:
                    self._probs[src_tag][trg_tag] += 1.
        sum_trg = defaultdict(float)
        for src_tag in self._probs:
            for trg_tag in self._probs[src_tag]:
                sum_trg[src_tag] += self._probs[src_tag][trg_tag]
            for trg_tag in self._probs[src_tag]:
                self._probs[src_tag][trg_tag] /= sum_trg[src_tag]
        
    def get_prior_prob(self, src_tag, trg_tag, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        prob = self._probs[src_tag][trg_tag]
        prob *= 1./(abs(1.*src_index/src_length - 1.*trg_index/trg_length) + 1)
        return prob

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        pass
        
    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass