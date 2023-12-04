import numpy as np

# # class SVM:
# # 	def __init__(self, learning_rate, max_iterations):

# import numpy as np
# class SVM:

# 	def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
# 		self.lr = learning_rate
# 		self.lambda_param = lambda_param
# 		self.n_iters = n_iters
# 		self.w = None 
# 		self.b = None
	
# 	def fit(self, X, y):
# 		n_samples, n_features = X.shape
# 		y_ = np.where(y <= 0, -1, 1)

# 		#init weights
# 		self.w= np.zeros(n_features)
# 		self.b = 0
# 		for _ in range(self.n_iters):
# 			for idx, x_i in enumerate(X):
# 				condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 
# 				if condition:
# 					self.w -= self.lr * (2 * self.lambda_param * self.w)
# 				else:
# 					self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
# 					self.b -= self.lr * y_[idx]

# 	def predict(self, X):
# 		approx = np.dot (X, self.w) - self.b
# 		return np.sign(approx)



# # Testing
# if __name__ == "__main__":
# 	# Imports
# 	from sklearn.model_selection import train_test_split
# 	from sklearn import datasets
# 	import matplotlib.pyplot as plt
# 	X, y = datasets.make_blobs(
# 		n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
# 	)
# 	y = np.where(y == 0, -1, 1)
# 	X_train, X_test, y_train, y_test = train_test_split(
# 		X, y, test_size=0.2, random_state=123
# 	)
	

# 	clf = SVM()
# 	clf.fit(X_train, y_train) 
# 	predictions = clf.predict(X_test)
# 	def accuracy (y_true, y_pred):
# 		accuracy = np.sum(y_true == y_pred) / len(y_true)
# 		return accuracy
# 	print("SVM classification accuracy", accuracy(y_test, predictions))



# 	def visualize_svm():
# 		def get_hyperplane_value(x, w, b, offset): 
# 			return (-w[0] *x + b + offset) / w[1]
		
# 		fig = plt.figure()
# 		ax = fig.add_subplot(1, 1, 1)
# 		plt.scatter (X[:, 0], X[:, 1], marker="o", c=y)
# 		x0_1 = np.amin (X[:, 0]) 
# 		x0_2 = np.amax (X[:, 0])
# 		x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0) 
# 		x1_2 = get_hyperplane_value (x0_2, clf.w, clf.b, 0)

# 		x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1) 
# 		x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

# 		x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1) 
# 		x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

		

# 		ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
# 		ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k") 
# 		ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")
# 		x1_min = np.amin (X[:, 1])
# 		x1_max = np.amax(X[:, 1])
# 		ax.set_ylim([x1_min - 3, x1_max + 3])
# 		plt.show()

# 	visualize_svm()

# class SVM:
#     def __init__(self, C=1.0):
#         self.C = C
#         self.alpha = None
#         self.b = None

#     def fit(self, X, y, epochs=100, tol=1e-3):
#         n_samples, n_features = X.shape

#         # Initialize alpha and b
#         self.alpha = np.zeros(n_samples)
#         self.b = 0.0

#         # SMO optimization
#         for epoch in range(epochs):
#             alpha_changed = 0
#             for i in range(n_samples):
#                 # Compute the predicted class
#                 f_i = self.predict(X[i, :])
#                 E_i = f_i - y[i]

#                 # Check if the sample violates the KKT conditions
#                 if (y[i] * E_i < -tol and self.alpha[i] < self.C) or (y[i] * E_i > tol and self.alpha[i] > 0):
#                     j = self.select_second_sample(i, n_samples)
#                     f_j = self.predict(X[j, :])
#                     E_j = f_j - y[j]

#                     # Save the current alpha values
#                     alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

#                     # Compute the bounds for alpha[j]
#                     L, H = self.compute_bounds(i, j, y)

#                     if L == H:
#                         continue

#                     # Compute the kernel
#                     K_ij = np.dot(X[i, :], X[j, :])
#                     K_ii = np.dot(X[i, :], X[i, :])
#                     K_jj = np.dot(X[j, :], X[j, :])

#                     # Update alpha[j]
#                     self.alpha[j] = alpha_j_old + y[j] * (E_i - E_j) / (K_ii + K_jj - 2 * K_ij)

#                     # Clip alpha[j] to be within the bounds [L, H]
#                     self.alpha[j] = np.clip(self.alpha[j], L, H)

#                     # Update alpha[i]
#                     self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])

#                     # Update the bias terms
#                     b_i_new = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K_ii - y[j] * (self.alpha[j] - alpha_j_old) * K_ij
#                     b_j_new = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K_ij - y[j] * (self.alpha[j] - alpha_j_old) * K_jj

#                     if 0 < self.alpha[i] < self.C:
#                         self.b = b_i_new
#                     elif 0 < self.alpha[j] < self.C:
#                         self.b = b_j_new
#                     else:
#                         self.b = (b_i_new + b_j_new) / 2

#                     alpha_changed += 1

#             # Check for convergence
#             if alpha_changed == 0:
#                 break

#     def predict(self, X):
#         # Calculate the decision function
#         decision_function = np.dot(self.alpha * y, np.dot(X, X_train.T)) + self.b
#         return np.sign(decision_function)

#     def select_second_sample(self, i, n_samples):
#         j = i
#         while j == i:
#             j = np.random.randint(0, n_samples)
#         return j

#     def compute_bounds(self, i, j, y):
#         if y[i] != y[j]:
#             L = max(0, self.alpha[j] - self.alpha[i])
#             H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
#         else:
#             L = max(0, self.alpha[i] + self.alpha[j] - self.C)
#             H = min(self.C, self.alpha[i] + self.alpha[j])
#         return L, H

# Example usage:
# Assume X_train is your feature matrix and y_train is your target vector (-1 or 1)
# You might need to normalize your data and preprocess it according to your needs.

# Example data
# X_train = np.array([[3, 4], [1, 2], [5, 6], [-1, -2], [-3, -4], [-5, -6]])
# y_train = np.array([1, 1, 1, -1, -1, -1])

# # Instantiate and fit the SVM
# svm = SVM(C=1.0)
# svm.fit(X_train, y_train)

# # Make predictions on new data
# X_new = np.array([[0, 0], [-2, -3], [4, 5]])
# predictions = svm.predict(X_new)

# print("Predictions:", predictions)

class SVM_Linear:
    
	def __init__(self, learning_rate=0.001, lambda_param=0.01, iters=1000):
		self.lr = learning_rate
		self.lambda_param = lambda_param
		self.iters = iters
		self.w1 = None
		self.w2 = None
		self.w3 = None
		self.b = None

	def train(self, X, y):
		# n_samples, n_features = X.shape
		# y_ = np.where(y <= 0, -1, 1)

		#init weights
		# self.w= np.zeros(n_features)
		self.w1= 0
		self.b = 0
		for _ in range(self.iters):
			for i, x_i in enumerate(X):
				# condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 
				condition = y[i] * (self.w1 * x_i[0] - self.b) >= 1 
				if condition:
					self.w1 -= self.lr * (2 * self.lambda_param * self.w1)
				else:
					self.w1 -= self.lr * (2 * self.lambda_param * self.w1 - x_i[0]*y[i])
					self.b -= self.lr * y[i]
# 				condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 
# 				if condition:
# 					self.w -= self.lr * (2 * self.lambda_param * self.w)
# 				else:
# 					self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
# 					self.b -= self.lr * y_[idx]

	def predict(self, X):
		approx = X[0] * self.w1 - self.b
		# approx = np.dot (X, self.w) - self.b
		return np.sign(approx)



import pandas as pd

# Read only the desired columns from the CSV file
columns_to_select = ["math score", "reading score", "writing score"]
df = pd.read_csv('train.csv', usecols=columns_to_select)
x_train = []
y_train = []
# Display the DataFrame with only the selected columns
for index, row in df.iterrows():
    # Access individual elements in the row
    math_score = row['math score']
    # reading_score = row['reading score']
    # writing_score = row['writing score']
    # ar.append(math_score)
    # # Do something with the scores (replace this with your processing logic)
    # print(f"Math: {math_score}, Reading: {reading_score}, Writing: {writing_score}")
    x_train.append([math_score])
    if math_score < 60:
        y_train.append(-1)
    else:
        y_train.append(1)

df = pd.read_csv('test.csv', usecols=columns_to_select)
x_test = []
# Display the DataFrame with only the selected columns
for index, row in df.iterrows():
    # Access individual elements in the row
    math_score = row['math score']
    # reading_score = row['reading score']
    # writing_score = row['writing score']
    # ar.append(math_score)
    # # Do something with the scores (replace this with your processing logic)
    # print(f"Math: {math_score}, Reading: {reading_score}, Writing: {writing_score}")
    x_test.append([math_score])
    # if math_score < 60:
    #     y_train.append(-1)
    # else:
    #     y_train.append(1)


svm = SVM_Linear()
svm.train(x_train, y_train)
y_test = []
for x in x_test:
	y_test.append(svm.predict(x))
	print(f"x = {x[0]}, y ={svm.predict(x)} ")
a = 1
