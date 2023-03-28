#import modules
import pyforest
import pickle

#load dataset
df = pd.read_csv(r"C:\Users\User\Documents\model\BankNote_Authentication.csv")

#read dataset
print(df.head())

#select dependent and independent variables
X = df[['variance', 'skewness', 'curtosis', 'entropy']]
Y = df['class']

#split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 50)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#instantiate model
classifier = RandomForestClassifier()

#fitting
classifier.fit(X_train, Y_train)

#convert to pickle file
pickle.dump(classifier, open('model.pkl', 'wb'))
