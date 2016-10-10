__author__ = 'osopova'

from imports import *

path_to_test = "../hackerrank-predict-email-opens-dataset/test_dataset.csv"
test_dataset = pd.read_csv(path_to_test)

path_to_model = './output/2016-09-04-21-42-13pickle.dat'

# load model and data in
model = pickle.load(open(path_to_model, "rb"))
predictions = model.predict(test_dataset)

np.savetxt(generate_unique_filename() + "-predictions.csv", predictions, delimiter=",")
