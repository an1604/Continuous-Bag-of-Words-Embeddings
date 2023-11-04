# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:09:03 2023

@author: adina
"""

"""Transfer Learning Using Pretrained Embeddings for
Document Classification using CNN """
# imports 
import re
import string
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


"""The Vocabulary"""
class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx}

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
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        """
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
    
    
    
"""SequenceVocabulary - to track the sequence of the context using 
    differrnt tokens: 
        The UNK token - allows the model to learn a representation
            for rare words so that it can accommodate words that it has 
            never seen at test time.
        The MASK token - serves as a sentinel for Embedding layers 
            and loss calculations when we have sequences of variable length
       The BEGIN­OF­SEQUENCE and END­OF­SEQUENCE tokens - 
           give the neural network hints about the sequence boundaries.
           
        For example : 
            SequenceVocabulary{
                'MASK' : 0,
                'UNK' : 1,
                'BEGIN­OF­SEQUENCE' : 2,
                'END­OF­SEQUENCE' : 3,
                'some-word' :4,
                'some-other-word' :5 }
"""
class SequenceVocabulary(Vocabulary):
       
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):
               super(SequenceVocabulary, self).__init__(token_to_idx)
               # Initializing the tokens 
               self._mask_token = mask_token
               self._unk_token = unk_token
               self._begin_seq_token = begin_seq_token
               self._end_seq_token = end_seq_token
               # add those tokens to the vocabulary
               self.mask_index = self.add_token(self._mask_token)
               self.unk_index = self.add_token(self._unk_token)
               self.begin_seq_index = self.add_token(self._begin_seq_token)
               self.end_seq_index = self.add_token(self._end_seq_token)
               
    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

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
           
"""NewsVectorizer"""
class NewsVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""    
    def __init__(self, title_vocab, category_vocab):
        self.title_vocab = title_vocab
        self.category_vocab = category_vocab
    
    def vectorize(self, title, vector_length=-1):
       """
       Args:
           title (str): the string of words separated by a space
           vector_length (int): an argument for forcing the length of index vector
       Returns:
           the vetorized title (numpy.array)
       """
       # keeping the begin seq indeces 
       indices = [self.title_vocab.begin_seq_index]
       # appending tokens for the end of the array to keep the sequance
       indices.extend(self.title_vocab.lookup_token(token) 
                       for token in title.split(" "))
       # adding the end seq token to the end 
       indices.append(self.title_vocab.end_seq_index)
       if vector_length < 0:
            vector_length = len(indices)

        # Generating the out vector 
       out_vector = np.zeros(vector_length, dtype=np.int64)
       out_vector[:len(indices)] = indices
       out_vector[len(indices):] = self.title_vocab.mask_index
       
       return out_vector
   
    @classmethod
    def from_dataframe(cls, news_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            news_df (pandas.DataFrame): the target dataset
            cutoff (int): frequency threshold for including in Vocabulary 
        Returns:
            an instance of the NewsVectorizer
        """
        # Initialize new vocabulary
        category_vocab = Vocabulary() 
        # adding tokens into 
        for category in sorted(set(news_df.category)):
            category_vocab.add_token(category)
       
        # Counting all the frequencies foreach word
        word_counts = Counter()
        for title in news_df.title:
           for token in title.split(" "):
               if token not in string.punctuation:
                   word_counts[token] += 1
       
        # Making the SequenceVocabulary 
        title_vocab = SequenceVocabulary()
        # appending into only the words that passed the cutoff
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)
        
        return cls(title_vocab, category_vocab)

"""The Dataset"""
class NewsDataset(Dataset):
    def __init__(self, news_df, vectorizer):
        """
        Args:
            news_df (pandas.DataFrame): the dataset
            vectorizer (NewsVectorizer): vectorizer instatiated from dataset
        """
        self.news_df = news_df
        self._vectorizer = vectorizer
        
        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, news_df.title)) + 2
        
        self.train_df = self.news_df[self.news_df['split'] == 'train']
        self.val_df = self.news_df[self.news_df['split'] == 'val']
        self.test_df = self.news_df[self.news_df['split'] == 'test']
        # Kepping the sizes
        self.train_size = len(self.train_df)
        self.validation_size = len(self.val_df)
        self.test_size = len(self.test_df)
        
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # Class weights
        class_counts = news_df.category.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.category_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)



    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            surname_csv (str): location of the dataset
        Returns:
            an instance of SurnameDataset
        """
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split=='train']
        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df))

    
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
        
        title_vector = self._vectorizer.vectorize(row.title, self._max_seq_length)
        category_index =  self._vectorizer.category_vocab.lookup_token(row.category)
        return {'x_data': title_vector,
                'y_target': category_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

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
        
        
        
        


"""The NewsClassifier Model"""
class NewsClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, 
                 pretrained_embeddings=None, padding_idx=0):
        """Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            filter_width (int): width of the convolutional kernels
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter 
            retrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided, 
            padding_idx (int): an index representing a null position
        """
        super(NewsClassifier, self).__init__()
        
        if pretrained_embeddings is None:

            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                   kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                   kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                   kernel_size=3),
            nn.ELU()
        )

        self._dropout_p = dropout_p
        # mlp classifier
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        # embed and permute so features are channels
        # rearrange the dimensions of a tensor 
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        
        features = self.convnet(x_embedded)
        
        # average and remove the extra dimension- avarge pooling 
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self._dropout_p)
        
        # mlp classifier to produce classification outputs
        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)
        
        return prediction_vector
        

"""Training Routine"""
def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

hidden_dim=100
num_channels=100
learning_rate=0.001
dropout_p=0.1
batch_size=128
num_epochs=100
device = 'cpu'
embedding_size=100
hidden_dim=100



dataset = NewsDataset.load_dataset_and_make_vectorizer('news_with_splits.csv')
vectorizer = dataset.get_vectorizer()
classifier = NewsClassifier(embedding_size =  embedding_size, 
                            num_embeddings = len(vectorizer.title_vocab), 
                            num_channels = num_channels,
                            hidden_dim = hidden_dim, 
                            num_classes=len(vectorizer.category_vocab),
                            dropout_p = dropout_p,
                            padding_idx= 0 )


"""Training loop"""
classifier = classifier.to(device)
dataset.class_weights = dataset.class_weights.to(device)

loss_func = nn.CrossEntropyLoss(dataset.class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)


try : 
    print('Running over the training set {} times to generate batches...'.format(str(num_epochs)))
    for epoch_index in range(num_epochs):
        dataset.set_split('train')
        batch_generator = generate_batches(dataset, 
                                          batch_size=batch_size, 
                                          device=device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
     
       
    print('Running over the training set foreach batch ...')
    for batch_index, batch_dict in enumerate(batch_generator):
        # step 1. zero the gradients
        optimizer.zero_grad()
        
        # step 2. compute the output
        y_pred = classifier(batch_dict['x_data'])

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
        y_pred =  classifier(x_in=batch_dict['x_data'])
        
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
    y_pred =  classifier(x_in=batch_dict['x_data'])
    
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



# Preprocess the reviews
def preprocess_text(text):
    # converts the text to lowercase, making all characters in the text lowercase.
    text = ' '.join(word.lower() for word in text.split(" "))
    # looks for specific punctuation marks like '.', ',', '!', and '?' in the text.
    text = re.sub(r"([.,!?])", r" \1 ", text)
    # removes characters that are not letters (a-zA-Z) or the specified punctuation marks (.,!?). It replaces them with a space.
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def predict_category(title, classifier, vectorizer, max_length):
    """Predict a News category for a new title
    
    Args:
        title (str): a raw title string
        classifier (NewsClassifier): an instance of the trained classifier
        vectorizer (NewsVectorizer): the corresponding vectorizer
        max_length (int): the max sequence length
            Note: CNNs are sensitive to the input data tensor size. 
                  This ensures to keep it the same size as the training data
    """
    title = preprocess_text(title)
    
    # Getting the title in tensor represntation afetr vectorizing him
    vectorized_title = torch.tensor(vectorizer.vectorize(title, vector_length=max_length))
    # using softmax beacuse we have multi class classification, and we want to get the probabailities foreach class 
    result = classifier(vectorized_title.unsqueeze(0), apply_softmax=True)
    # Keeping only 1 prediction (dim=1)
    probability_values, indices = result.max(dim=1)
    predicted_category = vectorizer.category_vocab.lookup_index(indices.item())
    
    return {'category' : predicted_category,
            'probability' : probability_values.item()}


def get_samples():
    samples = {}
    for cat in dataset.val_df.category.unique():
        samples[cat] = dataset.val_df.title[dataset.val_df.category==cat].tolist()[:5]
    return samples

val_samples = get_samples()

classifier = classifier.to("cpu")

for truth, sample_group in val_samples.items():
    print(f"True Category: {truth}")
    print("="*30)
    for sample in sample_group:
        prediction = predict_category(sample, classifier, 
                                vectorizer, dataset._max_seq_length + 1)
        print("Prediction: {} (p={:0.2f})".format(prediction['category'],
                                                  prediction['probability']))
        print("\t + Sample: {}".format(sample))
    print("-"*30 + "\n")





