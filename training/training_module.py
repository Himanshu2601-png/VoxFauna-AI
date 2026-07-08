from prepare_dataset import prepare_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



def train():
    X,y = prepare_dataset()
    print(X.shape)
    print(y.shape)
    