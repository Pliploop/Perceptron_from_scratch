import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score,zero_one_loss
import matplotlib.pyplot as plt
import seaborn as sns

class Perceptron:

    def __init__(self,n_epochs,early_stop=True,n_early_stop=10,training_step=1,shuffle=True):
        """
        paramet
        
        """


        self.n_epochs=n_epochs
        self.loss=np.empty(n_epochs)

        self.val_accuracy=np.empty(n_epochs)
        self.val_recall = np.empty(n_epochs)
        self.val_f1_score = np.empty(n_epochs)
        
        self.training_accuracy=np.empty(n_epochs)
        self.training_recall = np.empty(n_epochs)
        self.training_f1_score = np.empty(n_epochs)



        self.early_stop=early_stop
        self.n_early_stop = n_early_stop
        self.training_step = training_step
        self.shuffle = shuffle
        self.weights_list=[]


    def predict(self,X,training=False):
        if not training:
            if isinstance(X, pd.DataFrame):
                X_a = X.values
            else:
                X_a = X
            n_samples=X_a.shape[0]
            n_features= X_a.shape[1]
            X_a_2 = np.ones((n_samples,n_features+1))

            X_a_2[:,1:] = X_a
        else:
            X_a_2 = X
        temp = np.matmul(X_a_2,self.weights)
        
        return pd.Series(np.sign(temp))


    def fit(self,X_train,y_train,X_val=None,y_val=None):
        """
        parameters:
            X_train : training data features of size (n_features, n_training_samples)
            y_train : training data ground truth (of size n_features, n_validation_samples)
            X_val : validation set
        """
        # add initial bias to X and X_val
        n_features = X_train.values.shape[1]
        n_samples_train = X_train.values.shape[0]
        X_train_a = np.ones((n_samples_train,n_features+1))
        X_train_a[:,1:] = X_train.values
        X_train_a = np.reshape(X_train_a,(n_samples_train,n_features +1))
        
        y_train_a=y_train.values

        if X_val is not None:
            n_samples_val = X_val.values.shape[0]


            X_val_a = np.ones((n_samples_val,n_features+1))
            X_val_a[:,1:] = X_val.values
            X_val_a = np.reshape(X_val_a,(n_samples_val,n_features +1))
            y_val_a=y_val.values
        

        
        


        #initialize weights according to n_features:
        self.weights = np.random.rand(n_features+1)


        #we now have initialized weights and feature values. Training is just iterating over the epochs and over the samples

        for k in tqdm(range(self.n_epochs)):
            self.weights_list.append(self.weights)
            
            to_sum = np.zeros(X_train_a.shape)
            summed_test = np.zeros(n_features+1)

            # print(y_train_a.shape)
            # print(X_train_a.shape)
            # print(X_train_a[1].shape)

            for sample in range(n_samples_train):
                
                if y_train_a[sample]*np.inner(self.weights,X_train_a[sample])<0:
                    summed_test += y_train_a[sample]*X_train_a[sample]
            



            

            index_function = pd.Series(np.matmul(X_train_a,self.weights)*y_train).apply(lambda x:0 if x>=0 else 1).to_numpy()
            vector = index_function*y_train_a
            for i in range(n_samples_train):
                to_sum[i] = X_train_a[i]*vector[i]
            

            summed = to_sum.sum(axis=0)
            # print(summed_test==summed)

            self.weights = self.weights + self.training_step*summed


            self.training_accuracy[k]=accuracy_score(pd.Series(y_train_a),self.predict(X_train_a,training=True))
            self.loss[k]=zero_one_loss(pd.Series(y_train_a),self.predict(X_train_a,training=True))
            if X_val is not None:
                self.val_accuracy[k]=accuracy_score(pd.Series(y_val_a),self.predict(X_val_a,training=True))
            
        return self.weights
    
    

    def get_metrics(self):
        return self.training_accuracy,self.loss,self.val_accuracy

    def score(self,X,y):
        return accuracy_score(self.predict(X),y)

    def plot_metrics(self):
        fig,ax=plt.subplots(1,3,figsize=(15,5))
        ax[0].plot(self.training_accuracy,color='r')
        ax[0].plot(self.val_accuracy,color='b')
        ax[1].plot(self.loss,color='k')
        plt.show()

