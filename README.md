# Word Embedding Tasks
This repository includes code and resources related to word embedding tasks using Continuous Bag of Words (CBOW) and Pretrained Embeddings for Document Classification using Convolutional Neural Networks (CNN).

## Project 1: CBOW Word Embeddings with "The Frankenstein Dataset"
The first project is centered around the Continuous Bag of Words (CBOW) model, which is a type of word embedding model. The primary task of the CBOW model is multiclass classification. It involves scanning over text data, creating a context window of words, removing the center word from the context window, and classifying the context window to predict the missing word—a bit like a fill-in-the-blank task.

The key objective of this project is to introduce the use of the nn.Embedding layer, a PyTorch module that encapsulates an embedding matrix. This layer allows us to map a token's integer ID to a vector, which is used in neural network computations. During the training process, as the optimizer updates model weights to minimize loss, it also updates the values of the vector. This helps the model learn how to embed words effectively for the specific task.

In summary, Project 1 leverages the CBOW model to train word embeddings using "The Frankenstein Dataset" from Mary Shelley's novel "Frankenstein." The project encompasses data preprocessing, dataset creation, model training, and evaluation.

## Project 2: Document Classification Using CNN with Pretrained Word Embeddings
The second project focuses on document classification using Convolutional Neural Networks (CNNs) with pretrained word embeddings. This project tackles the task of classifying documents into different categories based on their content.

The project involves importing and processing text data for document classification. It also demonstrates the use of pretrained word embeddings to enhance the model's understanding of the textual content. By utilizing CNNs, the project showcases how convolutional layers can capture local features within the documents.

Project 2 provides insights into working with text data, incorporating pretrained embeddings, building a CNN-based classifier, and training the model for document classification tasks.

In both projects, word embeddings play a crucial role in understanding and representing the meaning of words and documents. While Project 1 primarily focuses on the fundamentals of word embeddings, Project 2 extends this knowledge to document-level tasks, showcasing the versatility of word embeddings in natural language processing.
