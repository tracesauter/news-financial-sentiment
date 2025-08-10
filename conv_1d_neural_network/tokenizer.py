# tokenizer for byte pair encoding created with Google Gemini
import unicodedata
import collections

def normalize_and_lowercase(text):
    # casts to lowercase and removes accents/diacritics.
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )

class BPETokenizer:
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab = []
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_pair_stats(self, tokenized_words):
        pairs = collections.defaultdict(int)
        for word_tokens in tokenized_words:
            for i in range(len(word_tokens) - 1):
                pairs[word_tokens[i], word_tokens[i+1]] += 1
        return pairs

    def _merge_pair(self, tokenized_words, pair, new_token):
        new_tokenized_words = []
        for word_tokens in tokenized_words:
            new_word = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            new_tokenized_words.append(new_word)
        return new_tokenized_words

    def train(self, text_corpus, vocab_size):
        normalized_corpus = [normalize_and_lowercase(text) for text in text_corpus]

        # Pre-tokenize to handle spaces: attach space to the start of each word but the first
        # this preserves word boundaries within the BPE algorithm.
        pre_tokenized_corpus = []
        for text in normalized_corpus:
            words = text.split()
            if not words: continue
            pre_tokenized_corpus.append(words[0])
            pre_tokenized_corpus.extend([f" {w}" for w in words[1:]])
        
        tokenized_words = [[char for char in word] for word in pre_tokenized_corpus]

        char_vocab = sorted(list(set(char for word in tokenized_words for char in word)))
        self.vocab = self.special_tokens + char_vocab
        
        if vocab_size < len(self.vocab):
            raise ValueError("vocab_size must be at least the size of special tokens + initial characters.")

        num_merges = vocab_size - len(self.vocab)
        for i in range(num_merges):
            pair_stats = self._get_pair_stats(tokenized_words)
            if not pair_stats: break
            
            best_pair = max(pair_stats, key=pair_stats.get)
            new_token = "".join(best_pair)
            
            tokenized_words = self._merge_pair(tokenized_words, best_pair, new_token)
            
            self.vocab.append(new_token)
            self.merges[best_pair] = new_token
            
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def encode(self, text):
        if not self.token_to_id:
            raise RuntimeError("Tokenizer has not been trained yet. Call train() first.")

        unk_token = "<UNK>"
        unk_id = self.token_to_id.get(unk_token)
        
        normalized_text = normalize_and_lowercase(text)
        
        # Apply the same pre-tokenization as in training
        words = normalized_text.split()
        if not words: return []
        
        pre_tokenized_words = [words[0]]
        pre_tokenized_words.extend([f" {w}" for w in words[1:]])
        
        encoded_ids = []
        for word in pre_tokenized_words:
            # For each word, perform a longest-match tokenization
            word_tokens = []
            i = 0
            while i < len(word):
                # Find the longest possible token from the vocab that starts at position i
                longest_match = ""
                # Iterate from the end of the word backwards
                for j in range(len(word), i, -1):
                    substring = word[i:j]
                    if substring in self.token_to_id:
                        longest_match = substring
                        break # found longest one. no need to check shorter ones
                
                if longest_match:
                    word_tokens.append(longest_match)
                    i += len(longest_match)
                else:
                    # If no match is found (not even a single character), it's an unknown.
                    # We add the character itself and let the ID lookup handle it.
                    word_tokens.append(word[i])
                    i += 1
            
            # Convert the found string tokens to their corresponding IDs
            word_ids = [self.token_to_id.get(token, unk_id) for token in word_tokens]
            encoded_ids.extend(word_ids)

        return encoded_ids

    def decode(self, ids):
        if not self.id_to_token:
            raise RuntimeError("Tokenizer has not been trained yet. Call train() first.")
        
        # join the tokens - which already acounts for spaces
        tokens = [self.id_to_token.get(i, "") for i in ids]
        return "".join(tokens)
