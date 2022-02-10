#from scipy.io.wavfile import read
import scipy.io.wavfile as wavf
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV
import time
import pandas as pd

class LoadData:
    def __init__(self, kernelNum = 10_000, features = "ppv, sum", dataset = None):
        self.kernelNum = kernelNum
        self.dataset = dataset
        self.features = features
        self.readFileNum = 0
        self.read_data_timing = 0
        self.generate_kernel_timing = 0
        self.apply_kernels_timing = 0
        self.score_timing = 0
        self.mean_accuracy  = 0
        self.audioMNIST_data= []
        self.audioMNIST_labels = []

    def preprocess_data(self, src):
        # loop over recordings


        total_files = 0

        read_data_start = time.perf_counter()
        
        for folder in os.listdir(src):
            i = 0
            for filepath in sorted(glob.glob(os.path.join(src,folder, "*.wav"))):
                if i%2==0:
                    # infer sample info from name， dig is label(0-9)
                    dig, _vp, _rep = filepath.rstrip(".wav").split("\\")[-1].split("_")
                    _fs, data = wavf.read(filepath)
                    
                    # reshape all wave files to fixed size
                    new_data = np.zeros(43953, dtype = np.float64)
                    new_data[:min(data.shape[0],43953)] = data[:min(data.shape[0],43953)]
                    self.audioMNIST_data.append(new_data)
                    #audioMNIST_data.append(np.array(data,dtype=np.float64))
                    self.audioMNIST_labels.append(int(dig))

                i+=1
                total_files+= 1
            #break
        
        read_data_end = time.perf_counter()
        read_data_timing = read_data_end-read_data_start
        print(f"Read data : {read_data_timing:0.4f} seconds\n")

        self.readFileNum = total_files
        self.read_data_timing = round(read_data_timing,4)
        #self.rocket(audioMNIST_data, audioMNIST_labels)

    def rocket(self):
        # split data
        X_training, X_test, Y_training, Y_test =  train_test_split(np.array(self.audioMNIST_data),np.array(self.audioMNIST_labels), test_size=0.20)


        print("generating kernels.....")
        generate_start = time.perf_counter()

        #kernels = generate_kernels(np.int64(X_training.shape[-1]), np.int64(10_000))
        kernels = generate_kernels(np.int64(X_training.shape[-1]), np.int64(self.kernelNum))

        generate_end = time.perf_counter()
        generate_timing = generate_end - generate_start
        print(f"Generate kernels : {generate_timing:0.4f} seconds\n")

        print("applying kernels....")
        apply_start = time.perf_counter()

        X_training_transform = apply_kernels(X_training, kernels)
        
        apply_end = time.perf_counter()
        apply_timing = apply_end - apply_start
        print(f"Apply kernels : {apply_timing:0.4f} seconds\n")

        print("training.....")
        train_start = time.perf_counter()

        classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        classifier.fit(X_training_transform, Y_training)

        train_end= time.perf_counter()
        train_timing = train_end - train_start
        print(f"Train : {train_timing:0.4f} seconds\n")

        # transform test set and return the mean accuracy
        print("transforming test set and scoring.....")
        score_start = time.perf_counter()

        X_test_transform = apply_kernels(X_test, kernels)
        mean_accuracy = classifier.score(X_test_transform, Y_test)


        score_end = time.perf_counter()
        score_timing = score_end - score_start
        print(f"Predict : {score_timing:0.4f} seconds\n")
        print("------------mean_accuracy: --------")
        print(mean_accuracy)

        # # transform test set and predict
        # print("predicting.....")
        # predict_start = time.perf_counter()

        # X_test_transform = apply_kernels(X_test, kernels)
        # predictions = classifier.predict(X_test_transform)

        # predict_end = time.perf_counter()
        # predict_timing = predict_end - predict_start
        # print(f"Predict : {predict_timing:0.4f} seconds\n")
        # print("------------predictions: --------")
        # print(predictions)

        self.generate_kernel_timing = round(generate_timing,4)
        self.apply_kernels_timing = round(apply_timing,4)
        self.score_timing = round(score_timing,4)
        self.mean_accuracy = round(mean_accuracy,4)
    
    def recordResult(self, fileName=None):
        results = pd.DataFrame(index = [self.dataset],
                       columns = ["kernelNum",
                                  "features",
                                  "readFileNum",
                                  "generate_kernel_timing",
                                  "apply_kernels_timing",
                                  "score_timing",
                                  "mean_accuracy"],
                       data = {
                            "kernelNum":[self.kernelNum],
                            "features": [self.features],
                            "readFileNum" : [self.readFileNum],
                            "generate_kernel_timing" : [self.generate_kernel_timing],
                            "apply_kernels_timing" : [self.apply_kernels_timing],
                            "score_timing" : [self.score_timing],
                            "mean_accuracy" : [self.mean_accuracy]
                       })
        results.index.name = "dataset"

        # initial
        if fileName is None:
            results.to_csv('results.csv', mode='a', index=True, header=True)

        # append to existing file
        else:
            results.to_csv(fileName, mode='a', index=True, header=False)

if __name__ == "__main__":
    #load_data_class = LoadData(kernelNum= 3_000, features="ppv", dataset="audioMNIST") #[Done] exited with code=0 in 3718.334 seconds
    load_data_class = LoadData(kernelNum= 10_000, features="ppv", dataset="audioMNIST") #[Done] exited with code=0 in 6441.554 seconds
    src =os.path.join(os.getcwd(), "data")
    # dst = "preprocessed_data"
    load_data_class.preprocess_data(src)
    load_data_class.rocket()
    #load_data_class.recordResult() # initial 
    load_data_class.recordResult("results.csv")
