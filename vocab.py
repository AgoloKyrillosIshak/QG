class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.total_words = 0
        self.unknown = '<UNK>'
        self.NUM = '<NUM>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.add_word(self.PAD, count=0)
        self.add_word(self.EOS, count=0)
        self.add_word(self.unknown, count=0)
        # self.add_word(self.NUM, count=0)
        with open('vocab.txt') as vocab_file:
            for word in vocab_file:
                self.add_word(word.rstrip('\n'))

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word

    def construct(self, words):
        for word in words:
            self.add_word(word)

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def encode_list(self, word_list):
        indices=[self.encode(word) for word in word_list]
        return indices

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)