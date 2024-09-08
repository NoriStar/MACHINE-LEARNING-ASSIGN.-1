import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize

#Extracting the data using read_excel, given the file could not be downloaded as a csv on my laptop
Spotify = pd.read_excel('SpotifyFeatures.csv.xlsx')

#We have to first clean the data:
# Were ony interested in the features that have to do with two genres: 'pop'/'cclassical'
new_Spotify = Spotify[Spotify["genre"].str.contains("Pop|Classical", case=False, na=False)]

# A lot of the values are taken as strings so we must select the columns that we want to convert to integers using pandas conversion function
new_Spotify.loc[:, 'loudness'] = pd.to_numeric(new_Spotify['loudness'], errors='coerce')
new_Spotify.loc[:, 'liveness'] = pd.to_numeric(new_Spotify['liveness'], errors='coerce')

# There are many unecessary rows that we must 'drop'
new_Spotify = new_Spotify.dropna(subset=['loudness', 'liveness'])

# Initialize arrays for different categories
Genre = []
Loudness = []
Liveness = []
Song = []


#Appending values in our columns of interest into the different arrays
for index, row in new_Spotify.iterrows():
    if row['genre'] == 'Pop':
        Genre.append(1)
    elif row['genre'] == 'Classical':
        Genre.append(0)
    Loudness.append(row['loudness'])
    Liveness.append(row['liveness'])
    Song.append(row['track_name'])

#We combine Loudness and Livness arrays intto a matrix and transpose it to put the features in rows instead of columns so they allign corectly with genre
X = np.array([Loudness, Liveness]).T
#Genre array (1 or 0):
y = np.array(Genre).reshape(-1, 1)
#printing the number of values in each array:

print("X:", len(X))
print("y:", len(Genre))

# Calculate the splitting the data to test and train data
split_index = int(len(X) * 0.8)

# Split the data into training and test sets
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# We make a zero array of zeros for our weight given we are working with matrices. The array shouls have the same number of rows as X and one column
W = np.zeros((X_train.shape[1], 1), dtype=np.float64)
b = 0

# Function z which is a linear combination explaining our data
def compute_z(X, W, b):
    return np.dot(X, W) + b

# Sigmoid function for logistic regression, to turn data into a sigmoid graph, with each probability being close to 1 or 0
def sigmoid(z_value):
    return 1 / (1 + np.exp(-z_value))

# Function describing the inaccuracy of predictions in order to improve weights using derivatives
def Loglikelihood(X, y, lr, iterations, log_interval=5000):
   #number of features and variables in x:
    m=X.shape[0]
    n=X.shape[1]

    # Initialize weights and bias
    W=np.zeros((n, 1), dtype=np.float64)
    b=0

    # creating a list which stores all the costs, should see a decrease in values
    costs = []

    # Using a for loop to run features and corresponding values through functions
    for i in range(iterations):
        z_value = compute_z(X, W, b)
        o = sigmoid(z_value)

        # Compute the log-likelihood (cost)
        cost = -(1 /m) * np.sum(y*np.log(o+1e-8) + (1-y) * np.log(1-o+1e-8))

        # Store cost values at specified intervals, so we can calculate different epochs
        if i % log_interval == 0:
            print(f"cost after {i} iteration is: {cost}")
            costs.append(cost)

        # Derrivatives of costs with respect to weights and y intercept
        dw = (1 / m)*np.dot(X.T, (o-y))
        db = (1 / m)*np.sum(o-y)

        #Updated values:
        W -= lr * dw
        b -= lr * db

    # Plot the cost function against epochs
   #For epoch weve just had a range form 0 to the last interration but then dividing this by a specified amount the 'batch'
   # Plot the cost function against epochs


    epochs = range(0, iterations, log_interval)
    plt.plot(epochs, costs)  # Ensure lengths match
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost Function over Epochs')
    plt.show()

    return W, b

#Calling function using our X and Y training values. Also setting a learning rate and number of interrations which we will later use also to calculate the epochs using baches as well
W, b = Loglikelihood(X_train, y_train, lr=0.03, iterations=90000, log_interval=100)
#extracting the final weights:
print("Final weights:", W)
print("Final bias/y intercept:", b)
#epoch= total_interations/batch


#Running our test functions through:

# Applying our final weights to test data:

# Trained weights and bias
W = np.array([[0.4112341], [-1.73111889]])
b = 4.816120186189744

z_test = compute_z(X_test,W , b)
sigmoid_output = sigmoid(z_test)

#Plot X_test against the sigmoid output
plt.scatter(X_test[:, 0], sigmoid_output, c=y_test.flatten())

plt.xlabel('X_test')
plt.ylabel('Sigmoid')
plt.title('Loudness and livness')
plt.show()




#Testing for accuracy of test results

#placing the actual genre values into an array for further calculation
y2 = np.array(y_test).reshape(-1, 1)
#placing the genres into an array in
k = np.array(sigmoid(z_test)).reshape(-1, 1)

#coverting the probabilities to 1 or 0 by rounding off:
predictions = (k > 0.5).astype(int)

#Obtaining the right predictions
correct_predictions = np.sum(predictions == y2)

# Total number of predictions
total_predictions = len(y2)

#The Difference is actually the values that are WRONGLY predicted.
#Caus the probabilities of the difference are rounded off and therefore the greter the value the further it is from beingright
#So right now the 'Difference variable represents the inaccuracy, so it must be subtracted by 1,
#To get what values were actually PREDICTED RIGHT
accuracy = (correct_predictions / total_predictions) * 100

print(f'The accuracy of our model for testing data is {accuracy:.2f}%')


#Testing for accuracy of training results
#Using the same process

y1 = np.array(y_train).reshape(-1, 1)
z_test=compute_z(X_train,W,b)
k = np.array(sigmoid(z_test)).reshape(-1, 1)

predictions0 = (k > 0.5).astype(int)

correct_predictions = np.sum(predictions0 == y1)
total_predictions = len(y1)

accuracy = (correct_predictions / total_predictions) * 100

print(f'The accuracy of our model for training data is {accuracy:.2f}%')

#Creating a confusion matrix which displays how many of each category we got correct.
# Convert y2 and predictions to 1D arrays
y_actual = pd.Series(y2.ravel(), name='Actual')  # True labels
y_predicted = pd.Series(predictions.ravel(), name='Predicted')  # Predicted labels

# Create and print the confusion matrix
conf_matrix = pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'])
print(conf_matrix)
