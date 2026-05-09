'''text_dataset_char.py
Functions to create, organize, and preprocess a character level text dataset
Daniel Lyu, Ariel Pan
CS 443: Bio-inspired Machine Learning
Project 4: Recurrent Neural Networks
'''
import pandas as pd
import tensorflow as tf

from text_util import clean_review, make_train_val_split

class CharLevelDataset:
    '''Object for organizing, storing, and preprecessing IMDd-like char level text datasets organized like so:

    [Document/Review] -> [list of chars]

    Preprocessing adopts a char-level model.
    '''
    def __init__(self, file_path, pad_char='\x00', start_char='\x02', end_char='\x03', verbose=True):
        '''CharLevelDataset constructor

        Parameters:
        -----------
        file_path: str.
            File path to the .csv dataset.
        pad_char: str.
            Char to use to pad the end of each sequence to make sure every char sequence is exactly `seq_len` long.
            NOTE: The default char is an "invisible" non-printing char.
        start_char: str.
            Char to use to mark the start of each REVIEW. Counts toward the `seq_len` sequence length.
            NOTE: The default char is an "invisible" non-printing char.
        end_char: str.
            Char to use to mark the end of each REVIEW (before any padding). Counts toward the `seq_len` sequence len.
            NOTE: The default char is an "invisible" non-printing char.
        verbose: bool.
            If False, turns off all debugging print outs.

        TODO: Set instance variable for the constructor parameters.
        '''
        self.file_path = file_path
        self.pad_char = pad_char
        self.start_char = start_char
        self.end_char = end_char
        self.verbose = verbose
        
        # KEEP THE FOLLOWING PLACEHOLDERS, THESE ALL SHOULD BE SET BY THE process METHOD
        self.corpus = None  # Corpus organized as: list of strs, which each str is an entire movie review
        self.vocab = None  # Unique chars in the corpus
        self.char2ind_map = None  # Maps char to int codes
        self.ind2char_map = None  # Maps int codes to chars

        self.seqs_x_train_str = None  # seqs of input chars of len `seq_len` in training set
        self.seqs_x_train_int = None  # seqs of input chars (int-coded) of len `seq_len` in training set
        self.seqs_x_val_str = None  # seqs of input chars of len `seq_len` in val set
        self.seqs_x_val_int = None  # seqs of input chars (int-coded) of len `seq_len` in val set

        self.seqs_y_train_str = None  # seqs of next/target chars of len `seq_len` in training set
        self.seqs_y_train_int = None  # seqs of next/target chars (int-coded) of len `seq_len` in training set
        self.seqs_y_val_str = None  # seqs of next/target chars of len `seq_len` in val set
        self.seqs_y_val_int = None  # seqs of next/target chars (int-coded) of len `seq_len` in val set

    def get_filepath(self):
        '''Get the filepath to the dataset .CSV data file.'''
        return self.file_path

    def get_start_char(self):
        '''Return the char used to denote the start of a review.'''
        return self.start_char

    def get_end_char(self):
        '''Return the char used to denote the end of a review.'''
        return self.end_char

    def get_pad_char(self):
        '''Return the char used to pad the end of a sequence that includes the end of a review.'''
        return self.pad_char

    def get_corpus(self):
        '''Returns the corpus: list of str, where each str is an entire movie review.'''
        return self.corpus

    def get_vocab(self):
        '''Get the vocabulary, the unique list of chars in the corpus + pad, start, and end tokens.'''
        return self.vocab

    def get_char2ind_map(self):
        '''Get dictionary that looks up a char index (int) by its string.'''
        return self.char2ind_map

    def get_ind2char_map(self):
        '''Get dictionary that uses a char int code to by its string.'''
        return self.ind2char_map

    def get_train_split_str(self):
        '''Returns the str-coded input seqs and target seqs that compose the training set.

        (This method has been provided for you and should not require modification.)

        Returns:
        --------
        Python list of str. len=N_train.
            Str coded input sequences in training set. Each sequence length `seq_len`.
        Python list of str. len=N_train.
            Str coded target sequences in training set. Each sequence length `seq_len`.
        '''
        return self.seqs_x_train_str, self.seqs_y_train_str

    def get_train_split_int(self):
        '''Returns the int-coded input seqs and target seqs that compose the training set.

        (This method has been provided for you and should not require modification.)

        Returns:
        --------
        tf.int32 tensor. shape=(N_train, seq_len).
            Int coded input sequences in training set.
        tf.int32 tensor. shape=(N_train, seq_len).
            Int coded target sequences in training set.
        '''
        return self.seqs_x_train_int, self.seqs_y_train_int

    def get_val_split_str(self):
        '''Returns the str-coded input seqs and target seqs that compose the val set.

        (This method has been provided for you and should not require modification.)

        Returns:
        --------
        Python list of int. len=N_val.
            Int coded input sequences in validation set. Each sequence length `seq_len`.
        Python list of int. len=N_val.
            Int coded target sequences in validation set. Each sequence length `seq_len`.
        '''
        return self.seqs_x_val_str, self.seqs_y_val_str

    def get_val_split_int(self):
        '''Returns the int-coded input seqs and target seqs that compose the val set.

        (This method has been provided for you and should not require modification.)

        Returns:
        --------
        tf.int32 tensor. shape=(N_val, seq_len).
            Int coded input sequences in validation set.
        tf.int32 tensor. shape=(N_val, seq_len).
            Int coded target sequences in validation set.
        '''
        return self.seqs_x_val_int, self.seqs_y_val_int

    def load(self, N_reviews):
        '''Loads the text dataset .CSV file and formats the data as a 1D Python list, where each item is a single review
        represented as a string. Example:

        [<review 1 str>, <review 2 str>, <review 3 str>, ...]

        Parameters:
        -----------
        N_reviews: int.
            Number of reviews to retrieve/return sequentially, starting from the first review.
            If the user passes in -1, retrieve ALL available reviews.

        Returns:
        --------
        Python list of str. len=N_reviews
            Retrieved reviews, with each whole review represented as a single string (see example above).

        NOTE: Your implementation should be exactly the same as `load` in `WordLevelDataset` from the SOM project,
        except you should remove non-printable characters and HTML tags from each review. Do this using the
        `clean_review` function provided in text_util.py.
        '''
        df = pd.read_csv(self.file_path)

        if N_reviews < 0:
            N_reviews = len(df)

        # Extract reviews as a list of string (1 str per review)
        # NOTE: loc upper bound IS INCLUSIVE. So we need upper bound -1
        reviews = list(df.loc[:N_reviews-1, 'review'])
        # Clean review of non-printable characters and HTML tags
        reviews = [clean_review(review) for review in reviews]
        return reviews

    def make_vocabulary(self, corpus):
        '''Define the vocabulary in the corpus (unique chars in corpus + pad/start/end tokens).

        Parameters:
        -----------
        corpus: Python list of str.
            Each review represented by a single str.

        Returns:
        -----------
        Python list of str. len=vocab_sz.
            List of unique chars in the corpus, along with the pad/start/end tokens.
            The order should be: [pad char, start char, end char, unique chars in ascending alphabetical order]

        TODO: Add the end, start, and pad tokens to the vocab.
        '''
        # 1. Find all unique characters in the actual text
        unique_chars = set(''.join(corpus))

        # 2. Remove the special tokens if they happen to be in the text 
        # to avoid duplicates in the final list.
        specials = {self.pad_char, self.start_char, self.end_char}
        unique_chars -= specials

        # 3. Sort the remaining unique characters alphabetically
        sorted_chars = sorted(list(unique_chars))

        # 4. Construct the final list in the required order:
        # [pad char, start char, end char, unique chars...]
        vocab = [self.pad_char, self.start_char, self.end_char] + sorted_chars

        return vocab

    def make_char2ind_mapping(self, vocab):
        '''Create dictionary that looks up the index (int) of the char in the vocabulary.
        Indices for each char are in the range [0, vocab_sz-1].

        Parameters:
        -----------
        vocab: list of str.
            Unique characters in corpus.

        Returns:
        -----------
        Python dictionary. Key,value pairs: string,int
            Map between char and int code

        NOTE: Looking at your text preprocessing code from the SOM project should be very helpful...
        '''
        return dict((word, i) for i, word in enumerate(vocab))


    def make_ind2char_mapping(self, vocab):
        '''Create dictionary that uses a index in the vocabulary to look up the actual char
        Indices for each char are in the range [0, vocab_sz-1].

        Parameters:
        -----------
        vocab: list of str.
            Unique characters in corpus.

        Returns:
        -----------
        Python dictionary. Key,value pairs: int,str
            Map between int and char code

        NOTE: Looking at your text preprocessing code from the SOM project should be very helpful...
        '''
        return dict((i, word) for i, word in enumerate(vocab))

    def make_sequences(self, corpus, seq_len, seq_overlap):
        '''Divvy up each review in the corpus into a set of input and target sequences of length `seq_len`.
        Each target sequence is the same as the input sequence, except they are shifted by 1 char to codify the target
        prediction for a char should be the next char in the review.

        The last `seq_overlap` chars in the previous sequence should be the 1st `seq_overlap` chars in the NEXT seq.
        Chars should only "spill over" from one sequence from another WITHIN THE SAME REVIEW ONLY, not ACROSS REVIEWS.

        Example (without start/end tokens): 2 review corpus = ['abcdef', '123456'], seq_len=4, seq_overlap=2

        seqs_x: [['abcd'], ['cdef'],     ['1234'], ['3456']]
        seqs_y: [['bcde'], ['def<PAD>'], ['2345'], ['456<PAD>']]

        In reality, we also insert <START> and <END> tokens around each review.
        Example (with start/end tokens): 2 review corpus = ['abcdefg', '1234'], seq_len=4, seq_overlap=2

        seqs_x: [['<START>abc'], ['bcde'], ['defg'],     ['fg<END><PAD>'],     ['<START>123'], ['234<END>']]
        seqs_y: [['abcd'],       ['cdef'], ['efg<END>'], ['g<END><PAD><PAD>'], ['1234'],       ['34<END><PAD>']]

        Parameters:
        -----------
        corpus: Python list of str.
            Each review represented by a single str.
        seq_len: int.
            The number of chars (T) in each input and target sequence.
        seq_overlap: int.
            The number of chars from the end of the prev sequence within a review that bleeds over to the start of the
            next review.

        Returns:
        --------
        Python list of str. len=N. Str coded input sequences.
            Each sequence length `seq_len`.
        Python list of str. len=N. Str coded target sequences.
            Each sequence length `seq_len`.


        NOTE:
        1. I suggest using string rather than list operations to process each review. You will be processing a lot of
        text, so keeping efficiency in mind is important (str operations can be much faster list operations).
        2. While you may use NumPy/TensorFlow, using core Python operations is enough for a fast and effective
        implementation.
        3. Don't forget to count the start and end tokens as chars that should belong to each review.
        4. Don't forget to pad by whatever amount that is necessary to make every sequence, even at the end of reviews,
        exactly `seq_len` chars long.
        '''
        seqs_x = []
        seqs_y = []

        lower_stride = seq_len - seq_overlap  # How much we update/shift the lower bound of window w/ overlap
        chunk_len = seq_len + 1  # size of proto seq we extract before offsetting into x and y

        # Process each review
        for r in range(len(corpus)):
            # Pad the review str with start/end token
            curr_review = ''.join([self.start_char, corpus[r], self.end_char])

            lower = 0
            while lower < len(curr_review):
                # Get upper bound position that we want, accounting for y extra char, but might be out of bounds
                target_upper = lower + chunk_len
                # Get valid upper
                valid_upper = min(target_upper, len(curr_review))

                # Extract current seq + 1
                curr_chunk = curr_review[lower:valid_upper]

                # Compute how much we need to pad
                pad_amount = chunk_len - len(curr_chunk)
                # Do the padding
                # curr_chunk_padded = curr_chunk + pad_amount*[self.pad_token]
                curr_chunk_padded = ''.join([curr_chunk, pad_amount * self.pad_char])

                seqs_x.append(curr_chunk_padded[:seq_len])
                seqs_y.append(curr_chunk_padded[1:seq_len+1])

                # Break if our new lower goes out of bounds
                if lower + lower_stride >= len(curr_review):
                    break

                # Update lower for next window
                lower += lower_stride

        return seqs_x, seqs_y

    def convert_str2int(self, seqs, char2ind):
        '''Converts the corpus from string-coded to int-coded chars.

        Parameters:
        -----------
        seqs: Python list of str.
            Sequences made from the corpus represented as: [<seq1>, <seq2>, <seq3>, ...], where each <seq> is a
            string of length `seq_len`.
        char2ind: Python dictionary. key,value pairs: str,int
            Mapping from char string to int index in the vocab.

        Returns:
        --------
        tf.int32 tensor. shape=(N_seqs, seq_len).
            Each sequence in `seqs`, where each element/token is represented by its int code.
        '''
        seqs_ints = []
        for seq in seqs:
            curr_sent_int_coded = [char2ind[token] for token in seq]
            seqs_ints.append(curr_sent_int_coded)
        seq_ints_tf = tf.constant(seqs_ints, dtype=tf.int32)
        return seq_ints_tf

    def process(self, N_reviews, seq_len, seq_overlap, prop_val=0.2):
        '''Gets and preprocesses the IMDb dataset appropriately for training a RNN.
        This is a wrapper function to automate the functions you have already written.

        Parameters:
        -----------
        N_reviews: int.
            Number of reviews to load from the file.
        seq_len: int.
            The number of chars (T) in each input and target sequence.
        seq_overlap: int.
            The number of chars from the end of the prev sequence within a review that bleeds over to the start of the
            next review.
        prop_val: float.
            Proportion of the reviews to reserve for the validation set (i.e. exclude from training set).

        TODO:
        1. Use your existing methods to preprocess the dataset.
        2. Assign ALL instance variables from constructor for quick retrieval of the dataset and related variables after
        the preprocessing completes.
        3. Use the provided `make_train_val_split` function in `text_util.py` to split the corpus (list of reviews, each
        represented as a str) into reviews that should form the basis for the train and validation sequences.
        '''
        # TODO: use existing methods to compute constructor instance variables you don't see already below

        # 1. Load the raw cleaned reviews into the corpus
        self.corpus = self.load(N_reviews)

        # 2. Build the vocabulary and the bidirectional mappings
        self.vocab = self.make_vocabulary(self.corpus)
        self.char2ind_map = self.make_char2ind_mapping(self.vocab)
        self.ind2char_map = self.make_ind2char_mapping(self.vocab)

        if self.verbose:
            print('Number of unique chars/tokens:', len(self.vocab))

        # 3. Subdivide the reviews into train/val split before making sequences
        # This ensures reviews are kept whole within their respective splits
        r_train, r_val = make_train_val_split(self.corpus, prop_val=prop_val)
        splits = {'train': r_train, 'val': r_val}

        for split in splits:
            # 4. Generate the overlapping sliding window sequences for this split
            seqs_x_str, seqs_y_str = self.make_sequences(splits[split], seq_len, seq_overlap)

            if self.verbose:
                print(f'Number of {split} sequences:', len(seqs_x_str), 'of length', seq_len)

            # 5. Convert the string sequences into TensorFlow integer tensors
            seqs_x_int = self.convert_str2int(seqs_x_str, self.char2ind_map)
            seqs_y_int = self.convert_str2int(seqs_y_str, self.char2ind_map)

            if self.verbose:
                print(f'  {seqs_x_int.shape=} {seqs_y_int.shape=}')

            # 6. Assign to instance variables based on the current split
            if split == 'train':
                self.seqs_x_train_int = seqs_x_int
                self.seqs_y_train_int = seqs_y_int
                self.seqs_x_train_str = seqs_x_str
                self.seqs_y_train_str = seqs_y_str
            elif split == 'val':
                self.seqs_x_val_int = seqs_x_int
                self.seqs_y_val_int = seqs_y_int
                self.seqs_x_val_str = seqs_x_str
                self.seqs_y_val_str = seqs_y_str
            else:
                raise ValueError('Unsupported split')
