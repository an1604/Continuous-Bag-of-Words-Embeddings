# Continuous Bag of Words Embeddings (CBOW) Learning

The CBOW model is a multiclass classification task represented by scanning over texts of words, creating a context window of words, removing the center word from the context window, and classifying the context window to the missing word. Intuitively, I can think of it like a fill-in-the-blank task. There is a sentence with a missing word, and the model’s job is to figure out what that word should be.

The goal of this example is to introduce the nn.Embedding layer, a PyTorch module that encapsulates an embedding matrix. Using the Embedding layer, I can map a token’s integer ID to a vector that is used in the neural network computation. When the optimizer updates the model weights to minimize the loss, it also updates the values of the vector. Through this process, the model will learn to embed words in the most useful way it can for that task.

In the remainder of this example, I follow the standard example format. In the first section, I introduce the dataset, Mary Shelley’s Frankenstein. Then, I discuss the vectorization pipeline from token to vectorized minibatch. After that, I outline the CBOW classification model and how the Embedding layer is used. Next, I cover the training routine (although if you have been reading the book sequentially, the training should be fairly familiar at this point). Finally, I discuss the model evaluation, model inference, and how I can inspect the model.

## The Frankenstein Dataset

For this example, I will build a text dataset from a digitized version of Mary Shelley’s novel Frankenstein, available via Project Gutenberg. This section walks through the preprocessing, building a PyTorch Dataset class for this text dataset, and finally splitting the dataset into training, validation, and test sets.

# Code Implementation
The code is organized into different sections:

 Vocabulary: Defines a class to process text and extract vocabulary for mapping tokens to indices.

 CBOWVectorizer: Implements the vectorizer that maps context words to vectors.

 CBOWDataset: Constructs a dataset class for the CBOW task, which includes methods for loading and processing data.

 CBOWClassifier: Defines the CBOW model for word embeddings and classification.

 Training Routine: Contains the training loop to train the CBOW model.

 Testing the Model: Evaluates the trained model on the test dataset.

 Trained Embeddings: Demonstrates how to use the learned word embeddings by finding similar words.

# Training the CBOW Model
The training process is divided into iterations over the training dataset, where batches of data are processed in each iteration. The model's loss is minimized, and the accuracy is computed. The training loop runs for a specified number of epochs.

# Testing the CBOW Model
After training, the model is tested on the test dataset, and the test loss and accuracy are reported.

# Trained Embeddings
You can use the trained word embeddings to find words that are semantically similar to a given target word. The code allows you to input a word and retrieve the closest words in terms of similarity. Additionally, you can explore the similarities for a list of target words such as "frankenstein," "monster," "science," and more.
