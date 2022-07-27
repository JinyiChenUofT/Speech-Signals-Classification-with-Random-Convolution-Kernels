import scipy.io.wavfile as wavf
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV
import time
import pandas as pd
from sendEmail import sendEmail
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class LoadData:
    def __init__(self, kernelNum = 10_000, features = "ppv, sum", dataset = None, percentageOfTrainData=1.0):
        self.kernelNum = kernelNum
        self.dataset = dataset
        self.features = features
        self.readFileNum = 0
        self.read_data_timing = 0
        self.generate_kernel_timing = 0
        self.apply_kernels_timing = 0
        self.score_timing = 0
        self.mean_accuracy  = 0
        self.data= []
        self.labels = []
        self.scores = []
        self.percentageOfTrainData = percentageOfTrainData

        
    def preprocess_data(self, src):
        # loop over recordings


        total_files = 0

        read_data_start = time.perf_counter()
        

        #i = 0
        for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
            #if i%2 == 0:
            fold, src_file, esc10, target = filepath.rstrip(".wav").split("\\")[-1].split("-")
            _fs, data = wavf.read(filepath)

            self.data.append(np.array(data,dtype=np.float64))
            self.labels.append(int(target))
            total_files+= 1

            #i += 1
        print(len(self.labels))

        c1 = Counter(self.labels)
        print("Labels : ",sorted(c1.items(), key=lambda pair: pair[0]))


        
        read_data_end = time.perf_counter()
        read_data_timing = read_data_end-read_data_start
        print(f"Read data : {read_data_timing:0.4f} seconds\n")

        self.readFileNum = total_files
        self.read_data_timing = round(read_data_timing,4)


    def rocket(self):
        # split data 20% for test and 80% for training
        X_training, X_test, Y_training, Y_test =  train_test_split(np.array(self.data),np.array(self.labels), test_size=0.20)
        

        accuracy_scores = []
        mse_list = []
        cv = KFold(n_splits=10, random_state=None, shuffle=True)
        i = 0
        for train_index, test_index in cv.split(X_training):
            print("Train Index: ", train_index, "\n")
            print("Test Index: ", test_index)

            train_index = np.random.choice(train_index, int(self.percentageOfTrainData*len(train_index)), replace=False)
            x_train, y_train = X_training[train_index], Y_training[train_index]

            
            print(x_train[0])
            print(y_train[0])
            print(x_train.shape)
            print(x_train.shape[0])

            print(y_train.shape)
            print(y_train.shape[0])

            print("generating kernels.....")
            kernels = generate_kernels(np.int64(x_train.shape[-1]), np.int64(self.kernelNum))

            print("applying kernels....")
            x_train = apply_kernels(x_train, kernels)
            

            print("training.....")
            classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
            classifier.fit(x_train, y_train)

            # transform test set and return the mean accuracy
            print("transforming test set and scoring.....")
            X_test_transform = apply_kernels(X_test, kernels)
            Y_predict = classifier.predict(X_test_transform)
            mean_accuracy = classifier.score(X_test_transform, Y_test)
            print("------------ mean_accuracy: --------")
            print(mean_accuracy)
            
            accuracy_scores.append(round(mean_accuracy,4))

            mse = self.mean_squared_error(Y_test, Y_predict)
            mse_list.append(mse)
            confusion = confusion_matrix(Y_test, Y_predict)
            ax = sns.heatmap(confusion/np.sum(confusion), annot=True,
                        fmt='.2%', cmap='Blues')


            ax.set_title(str(self.kernelNum)+'kernel_'+str(self.readFileNum)+'files_'+self.features+'\n')


            # Display the visualization of the Confusion Matrix.
            #plt.show()
            plt.savefig(str(self.kernelNum)+'kernel_'+str(self.readFileNum)+'files_'+self.features+str(i))
            i += 1

        print("------------accuracy scores: --------")
        print(accuracy_scores)
        average_mean_accuracy = sum(accuracy_scores)/len(accuracy_scores)
        print("------------average mean_accuracy: --------")
        print(average_mean_accuracy)
        self.mean_accuracy = round(average_mean_accuracy,4)
        average_mse = sum(mse_list)/len(mse_list)
        self.average_mse = round(average_mse,4)
    
    def mean_squared_error(self, act, pred):

        diff = pred - act
        differences_squared = diff ** 2
        mean_diff = differences_squared.mean()

        return mean_diff

    def recordResult(self, fileName=None):
        results = pd.DataFrame(index = [self.dataset],
                       columns = ["kernelNum",
                                  "features",
                                  "readFileNum",
                                  "mean_accuracy",
                                  "percentageOfTrainData",
                                  "mse"],
                       data = {
                            "kernelNum":[self.kernelNum],
                            "features": [self.features],
                            "readFileNum" : [self.readFileNum],
                            "mean_accuracy" : [self.mean_accuracy],
                            "percentageOfTrainData" : [self.percentageOfTrainData],
                            "mse":[self.average_mse]
                       })
        results.index.name = "dataset"

        # initial
        if fileName is None:
            results.to_csv('results.csv', mode='a', index=True, header=True)

        # append to existing file
        else:
            results.to_csv(fileName, mode='a', index=True, header=False)

if __name__ == "__main__":
    load_data_class = LoadData(kernelNum= 1_000, dataset="ESC-50", percentageOfTrainData=0.5) 
    #load_data_class = LoadData(kernelNum= 5_000, features="ppv", dataset="ESC-50") 
    src =os.path.join(os.getcwd(), "ESC-50\\audio")

    load_data_class.preprocess_data(src)
    load_data_class.rocket()
    #load_data_class.recordResult() # initial 
    load_data_class.recordResult("results.csv")
    #sendEmail()