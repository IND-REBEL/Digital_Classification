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