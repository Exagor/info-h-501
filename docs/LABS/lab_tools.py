import numpy as np
import os
from sklearn.metrics import confusion_matrix
from skimage import draw
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import accuracy_score

class CIFAR10:    
    def __init__(self, path):
        self.path = path
        # Pre-load all data
        self.train = {}
        self.test = {}
        print('Pre-loading training data')
        self.train['images'] = np.load(os.path.join(path, 'images.npy')).astype('uint8')
        self.train['hog'] = np.load(os.path.join(path, 'images_hog.npy'))
        self.train['labels'] = np.load(os.path.join(path, 'labels_.npy')).astype('uint8')
        print('Pre-loading test data')
        self.test['images'] = np.load(os.path.join(path, 'test_images.npy')).astype('uint8')
        self.test['hog'] = np.load(os.path.join(path, 'test_images_hog.npy'))
        self.test['labels'] = np.load(os.path.join(path, 'test_labels.npy')).astype('uint8')
        
        self.labels = ['Airplane', 'Bird', 'Horse']

class CIFAR10_complete:
    def __init__(self, path):
        self.path = path
        # Pre-load all data
        self.train = {}
        self.test = {}
        print('Pre-loading training data')
        self.train['images'] = np.load(os.path.join(path, 'images.npy')).astype('uint8')
        self.train['hog'] = np.load(os.path.join(path, 'images_hog.npy'))
        self.train['labels'] = np.load(os.path.join(path, 'labels.npy')).astype('uint8')
        print('Pre-loading test data')
        self.test['images'] = np.load(os.path.join(path, 'test_images.npy')).astype('uint8')
        self.test['hog'] = np.load(os.path.join(path, 'test_images_hog.npy'))
        self.test['labels'] = np.load(os.path.join(path, 'test_labels.npy')).astype('uint8')
        
        self.labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer','Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def evaluate_classifier(clf, test_data, test_labels):
    pred = clf.predict(test_data)
    C = confusion_matrix(pred, test_labels)
    return C.diagonal().sum()*100./C.sum(),C


def get_hog_image(hog, output_size):
    orientations = 16
    cells = 4
    pxcell = output_size//cells
    
    radius = pxcell//2 - 1
    orientations_arr = np.arange(orientations)
    orientation_bin_midpoints = (np.pi * (orientations_arr + .5) / orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    if( hog.min() < 0 ): # If we are looking at neural network weights
        hog_image = np.zeros((cells*pxcell, cells*pxcell,3), dtype=float)
        norm = np.abs(hog).max()
    else: # If we are looking at actual HoGs from the dataset
        hog_image = np.zeros((cells*pxcell, cells*pxcell), dtype=float)
    for r in range(cells):
        for c in range(cells):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * pxcell + pxcell // 2,
                                c * pxcell + pxcell // 2])
                rr, cc = draw.line(int(centre[0] - dc),
                                   int(centre[1] + dr),
                                   int(centre[0] + dc),
                                   int(centre[1] - dr))
                if hog.min() < 0:
                    if( hog[r,c,o] >= 0 ):
                        hog_image[rr, cc, 2] += hog[r, c, o]/norm
                    else:
                        hog_image[rr, cc, 0] -= hog[r, c, o]/norm
                else:
                    hog_image[rr, cc] += hog[r, c, o]
    return hog_image


def find_best_hyperparameters(model, param_grid, X_train, y_train):
    """
    Function to find the best hyperparameters for a given model.
    Return the best model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    print("Searching best parameters...")
    grid_search.fit(X_train, y_train)
    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    return grid_search.best_params_

def evaluate_parameter(model_class, param_name, param_range, X_train, y_train, fixed_params=None, plot=True):
    """
    Evaluate a model's performance over a range of a specific parameter.

    Args:
        model_class: The class of the model (e.g., DecisionTreeClassifier).
        param_name: The name of the parameter to vary (e.g., 'max_depth').
        param_range: The range of values for the parameter (e.g., range(4, 20, 2)).
        X_train: Training data features.
        y_train: Training data labels.
        fixed_params: A dictionary of parameters to keep fixed (e.g., {'random_state': 42}).

    Returns:
        None. Displays a plot of accuracy vs parameter values.
    """
    accuracies = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for param_value in param_range:
        # Combine fixed parameters with the parameter being varied
        params = {**(fixed_params or {}), param_name: param_value}
        # Initialize the model with the specified parameters
        model = model_class(**params)
        # Evaluate on the test set
        accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        accuracies.append(accuracy)

    # Plot the results
    if plot:
        plt.figure()
        plt.plot(param_range, accuracies, marker='o')
        plt.title(f'Accuracy in function of {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')
        plt.xticks(param_range)
        plt.grid()
        plt.show()

    return accuracies

def ML_pipeline(model, X_train, y_train, X_test, y_test, plot_curves=False):
    """
    ML pipeline for training and testing a model.
    """

    print("=====================================")
    print("Descriptive Performance Metrics")
    print("=====================================")
    # cross-validation (how well it should perform on unseen data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') #it doesn't train the original model
    print("Cross-validation is done only on the training set")
    print("Accuracy from cross-validation :", scores)
    print("Cross-validation mean accuracy : ", scores.mean())
    # accuracy on training set
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    print("Training set accuracy (cheating) : ", accuracy_score(y_train,y_pred_train))
    # confusion matrix on training set
    cm_train = confusion_matrix(y_train, y_pred_train)
    print("Confusion Matrix (on training set) :\n",cm_train)

    print("=====================================")
    print("Predictive Performance Metrics")
    print("=====================================")
    y_pred = model.predict(X_test)
    # accuracy on test set (the performance on unseen data)
    print("Test set accuracy : ", accuracy_score(y_test, y_pred))
    # confusion matrix on test set
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (on test set) :\n",cm)
    print("Classification report :\n",classification_report(y_test, y_pred, digits=3)) 

    # learning curve
    if plot_curves:
        print("Learning Curve :")
        LearningCurveDisplay.from_estimator(
            model, X_train, y_train, train_sizes=[500, 1000, 1500, 2000, 2500, 3000], cv=cv)

    return scores.mean(), accuracy_score(y_test, y_pred), cm

