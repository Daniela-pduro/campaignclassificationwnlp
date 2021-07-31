# Directories Handle
import os


# Neural Network
from keras import models, layers

# Plotting
import matplotlib.pyplot as plt


class DeepLearningModelCreator(object):
    """
    This Python class takes an embedding matrix with weights, and max_length as input, and:
    
    - Builds a neural network containing an embedding layer.
    
    """
    
    # Initializer
    def __init__(self, embedding_matrix):
        
        self.embedding_matrix = embedding_matrix
        
    
    # Instance Methods   
    def model_builder(self, nclasses, length):
        """
        - Builds a neural network.
        """
        
        m = models.Sequential()
        
        # Embedding layer
        m.add(layers.Embedding(input_dim=self.embedding_matrix.shape[0], output_dim=self.embedding_matrix.shape[1], input_length=length, weights=[self.embedding_matrix], trainable=False))
        
        # Hidden layers
        m.add(layers.Dense(85, activation='relu'))
        m.add(layers.Dense(42, activation='relu'))
        m.add(layers.Dense(21, activation='relu'))
        
        # Flatten
        m.add(layers.Flatten())
              
        # Output layer
        m.add(layers.Dense(nclasses, activation='softmax'))
        
        return m
    
    def plot_metric(self, history, metric):
        
        history_dict = history.history
        values = history_dict[metric]
        
        if 'val_' + metric in history_dict.keys():
            
            val_values = history_dict['val_' + metric]
            
        epochs = range(1, len(values) + 1)
        
        if 'val_' + metric in history_dict.keys():
            
            plt.plot(epochs, val_values, label='Validation')
            
        plt.semilogy(epochs, values, label='Training')
        
        
        if 'val_' + metric in history_dict.keys():
            
            plt.title('Training and validation %s' % metric)
            
        else:
            
            plt.title('Training %s' % metric)
            
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()
        
        plt.show()