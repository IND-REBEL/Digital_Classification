from sklearn import svm ,datasets
from sklearn.model_selection import train_test_split

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y

def preprocess_data(data):
  # flatten the images
  n_samples = len(data)
  data = data.reshape((n_samples, -1))
  return data


# Split data into 50% train and 50% test subsets

def split_data(X, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


#train model of choise 
def train_model(x,y,model_params, model_type="svm"):
    if model_type == "svm":
        clf= svm.SVC
    model = clf(**model_params)
    model.fit(x,y)
    return model
    
    
    
def train_test_dev_split(X, y, test_size=0.5, dev_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_dev, X_test, y_dev, y_test = split_data(X_test, y_test, dev_size, random_state)
    return X_train, X_test, y_train, y_test, X_dev, y_dev

def predict_and_eval(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report
