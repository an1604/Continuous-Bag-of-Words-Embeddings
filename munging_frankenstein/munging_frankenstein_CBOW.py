# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:49:23 2023

@author: adina
"""

"""Learning the Continuous Bag of Words Embeddings (CBOW):"""
# imports
import re
import string

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Vocabulary
class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            mask_token (str): the MASK token to add into the Vocabulary; indicates
                a position that will not be used in updating the model's parameters
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
            
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token
        
        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 
        
    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx, 
                'add_unk': self._add_unk, 
                'unk_token': self._unk_token, 
                'mask_token': self._mask_token}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
            
    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary
        
        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


# The Vectorizer class
class CBOWVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""    
    def __init__(self, cbow_vocab):
        """
        Args:
            cbow_vocab (Vocabulary): maps words to integers
        """
        self.cbow_vocab = cbow_vocab
    
    def vectorize(self, context, vector_length=-1):
        """Args: 
            context (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        """    
        # getting the indeces foreach word in the context
        indices = [self.cbow_vocab.lookup_token(token) for token in context.split(' ')]
        
        if vector_length <0:
            vector_length = len(indices)
        
        """ out_vector is the vector of integers representing 
        the indices of the context is constructed and returned
        """
        out_vector = np.zeros(vector_length,dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector [len(indices):] = self.cbow_vocab.mask_index
        return out_vector 
    
    @classmethod
    def from_dataframe(cls, cbow_df):
       """Instantiate the vectorizer from the dataset dataframe
       
       Args:
           cbow_df (pandas.DataFrame): the target dataset
       Returns:
           an instance of the CBOWVectorizer
       """
       cbow_vocab = Vocabulary()
       for index, row in cbow_df.iterrows():
           for token in row.context.split(' '):
               cbow_vocab.add_token(token)
           cbow_vocab.add_token(row.target)
           
       return cls(cbow_vocab)

    @classmethod
    def from_serializable(cls, contents):
        cbow_vocab = \
            Vocabulary.from_serializable(contents['cbow_vocab'])
        return cls(cbow_vocab=cbow_vocab)

    def to_serializable(self):
        return {'cbow_vocab': self.cbow_vocab.to_serializable()}

# Constructing a dataset class for the CBOW task
class CBOWDataset(Dataset):
    def __init__(self, cbow_df, vectorizer):
        """Args:
            cbow_df (pandas.DataFrame): the dataset
            vectorizer (CBOWVectorizer): vectorizer instatiated from dataset"""
        self.cbow_df =cbow_df
        self._vectorizer = vectorizer
        
        # The measuring len - foreach line we splitting to tokens at first
        measure_len = lambda contex: len(contex.split(' '))
        self._max_seq_length = max(map(measure_len, cbow_df.context))
        
        #  splitting the data to 3 - train , validation and test 
        # np.random.seed(0)
        # self.cbow_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(self.cbow_df), p=[0.7, 0.15, 0.15])
        # Split the data into separate DataFrames
        self.train_df = self.cbow_df[self.cbow_df['split'] == 'train']
        self.val_df = self.cbow_df[self.cbow_df['split'] == 'val']
        self.test_df = self.cbow_df[self.cbow_df['split'] == 'test']
        # Kepping the sizes
        self.train_size = len(self.train_df)
        self.validation_size = len(self.val_df)
        self.test_size = len(self.test_df)
        
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')
    
    @classmethod
    def load_dataset_and_make_vectorizer(cls, cbow_csv):
       """Load dataset and make a new vectorizer from scratch.
       Args:
           cbow_csv (str): location of the dataset
       Returns:
           an instance of CBOWDataset
       """
       cbow_df = pd.read_csv(cbow_csv)
       np.random.seed(0)
       train_cbow_df =  cbow_df[cbow_df['split'] == 'train']
       return cls(cbow_df, CBOWVectorizer.from_dataframe(train_cbow_df))
   
    
    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer
    
    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        # vectorize the contex
        contex_vector = self._vectorizer.vectorize(row.context, self._max_seq_length)
        # getting the token index from the vectorizer instance
        target_index = self._vectorizer.cbow_vocab.lookup_token(row.target)
        
        return {'x_data':contex_vector,
                'y_target': target_index}
    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

# outsider function to generate the batches for the training process
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
            

# The Model: CBOW
class CBOWClassifier(nn.Module): # Simplified cbow Model    
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        """Args:
           vocabulary_size (int): number of vocabulary items, controls the
               number of embeddings and prediction vector size 
           embedding_size (int): size of the embeddings
           padding_idx (int): default 0; Embedding will not use this index 
        """
        super(CBOWClassifier, self).__init__()
        # The embedding layer
        self.embedding = nn.Embedding(num_embeddings =vocabulary_size, 
                                      embedding_dim = embedding_size,
                                      padding_idx= padding_idx)
        # The fully connected layer at the end the get the result
        self.fc1 = nn.Linear(in_features = embedding_size, 
                             out_features=vocabulary_size)
        # Keeping the size of the input & output for later checking 
        self.input_dim = embedding_size
        self.output_dim = vocabulary_size
        
        
    def forward(self, x_in, batch_size ,apply_softmax=False):
        """The forward pass of the classifier.
        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, input_dim)
            batch_size : the batch size to check x_in
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim).
        """
        # checking the shape of x_in to resist errors
        if x_in.shape[0] != batch_size and x_in.shape[1] != self.input_dim:
            raise Exception('x_in.shape is not (batch, input_dim)...')
        x_embedded_sum = self.embedding(x_in).sum(dim=1)
        y_out = self.fc1(x_embedded_sum)
        if apply_softmax :
            y_out = F.softmax(y_out)
        
        return y_out
        
        
        
"""The Training Routine"""        
def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

# Initialization
embedding_size=50
num_epochs=100
learning_rate=0.0001
batch_size=32
device = 'cpu'

dataset = CBOWDataset.load_dataset_and_make_vectorizer('frankenstein_with_splits.csv')        
vectorizer = dataset.get_vectorizer()
classifier = CBOWClassifier(vocabulary_size=len(vectorizer.cbow_vocab), 
                            embedding_size=embedding_size)
if classifier:
    print('classifier succesfullt created!')
        
"""Training loop"""        
classifier = classifier.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

dataset.set_split('train')
dataset.set_split('val')

try:
    print('Running over the training set num_epochs times...')
    for epoch_index in range(num_epochs):
        dataset.set_split('train')
        # Generating the batches
        batch_generator = generate_batches(dataset, 
                                           batch_size=batch_size, 
                                           device=device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        
    print('DONE!')
    
    print('Iterate over train dataset...')
    for batch_index, batch_dict in enumerate(batch_generator):
        # step 1. zero the gradients
        optimizer.zero_grad()
           
        # step 2. compute the output
        y_pred = classifier(x_in=batch_dict['x_data'] , batch_size = batch_size)
        
        # step 3. compute the loss
        y_target = batch_dict['y_target']
        loss = loss_func(y_pred,y_target)
        loss_t = loss.item()
        running_loss += (loss_t-running_loss)/(batch_index+1)
        
        # step 4. use loss to produce gradients
        loss.backward()
        
        # step 5. use optimizer to take gradient step
        optimizer.step()
        
        # compute the accuracy
        acc_t =compute_accuracy(y_pred, y_target)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        
    print('DONE!')
    print("Train loss: {};".format(running_loss))
    print("Train Accuracy: {}".format(running_acc))
    
    # Iterate over val dataset
    print('Iterate over val dataset...')
    # setup: batch generator, set loss and acc to 0; set eval mode on
    dataset.set_split('val')
    batch_generator = generate_batches(dataset, 
                                       batch_size=batch_size, 
                                       device=device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    
    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred =  classifier(x_in=batch_dict['x_data'], batch_size = batch_size)
        
        # step 3. compute the loss
        y_target = batch_dict['y_target']
        loss = loss_func(y_pred,y_target)
        loss_t = loss.item()
        running_loss += (loss_t-running_loss)/(batch_index+1)
        
        # compute the accuracy
        acc_t = compute_accuracy(y_pred, y_target)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        
       
    print('DONE!')
    print("Validation loss: {};".format(running_loss))
    print("Validation Accuracy: {}".format(running_acc))
        
except KeyboardInterrupt:
    print("Exiting loop")
    

"""Testing the model"""
classifier = classifier.to(device)
loss_func = nn.CrossEntropyLoss()
dataset.set_split('test')
batch_generator = generate_batches(dataset, 
                                   batch_size=batch_size, 
                                   device=device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred =  classifier(x_in=batch_dict['x_data'] ,batch_size = batch_size)
    
    # step 3. compute the loss
    y_target = batch_dict['y_target']
    loss = loss_func(y_pred,y_target)
    loss_t = loss.item()
    running_loss += (loss_t-running_loss)/(batch_index+1)
            
    # compute the accuracy
    acc_t = compute_accuracy(y_pred, y_target)
    running_acc += (acc_t - running_acc) / (batch_index + 1)
    
print('DONE!')
print("Test loss: {};".format(running_loss))
print("Test Accuracy: {}".format(running_acc))


"""Trained Embeddings"""
def pretty_print(results):
    """
    Pretty print embedding results.
    """
    for item in results:
        print ("...[%.2f] - %s"%(item[1], item[0]))


def get_closest(target_word, word_to_idx, embeddings, n=5):
    """
    Get the n closest
    words to your word.
    """

    # Calculate distances to all other words
    
    word_embedding = embeddings[word_to_idx[target_word.lower()]]
    distances = []
    for word, index in word_to_idx.items():
        if word == "<MASK>" or word == target_word:
            continue
        distances.append((word, torch.dist(word_embedding, embeddings[index])))
    
    results = sorted(distances, key=lambda x: x[1])[1:n+2]
    return results

word = input('Enter a word: ')
embeddings = classifier.embedding.weight.data
word_to_idx = vectorizer.cbow_vocab._token_to_idx
pretty_print(get_closest(word, word_to_idx, embeddings, n=5))


target_words = ['frankenstein', 'monster', 'science', 'sickness', 'lonely', 'happy']

embeddings = classifier.embedding.weight.data
word_to_idx = vectorizer.cbow_vocab._token_to_idx

for target_word in target_words: 
    print(f"======={target_word}=======")
    if target_word not in word_to_idx:
        print("Not in vocabulary")
        continue
    pretty_print(get_closest(target_word, word_to_idx, embeddings, n=5))




            
            
            
            
            
            
            
            
            
