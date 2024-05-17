import os
import glob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm, naive_bayes
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# locations
folder_paths = [
    "txt_sentoken/pos/folder1",
    "txt_sentoken/pos/folder2",
    "txt_sentoken/pos/folder3",
    "txt_sentoken/pos/folder4",
    "txt_sentoken/pos/folder5"
]

# all positive data
output_file = "combined_text_for_pos_sen.txt"

with open(output_file, 'w') as outfile:
    for folder_path in folder_paths:
        # Find all text files in the folder
        file_pattern = os.path.join(folder_path, "*.txt")
        file_list = sorted(glob.glob(file_pattern))  # Sort the file list

        for file_path in file_list:
            with open(file_path, 'r') as file:
                text = file.read()
                outfile.write(text)
                outfile.write("\n")

# positive is done

folder_paths_negative_sentiment = [
    "txt_sentoken/neg/folder1",
    "txt_sentoken/neg/folder2",
    "txt_sentoken/neg/folder3",
    "txt_sentoken/neg/folder4",
    "txt_sentoken/neg/folder5"
]
# all negative data
output_file_negative_sentiment = "combined_text_for_neg_sen.txt"

with open(output_file_negative_sentiment, 'w') as outfile:
    for folder_path in folder_paths_negative_sentiment:
        # Find all text files in the folder
        file_pattern = os.path.join(folder_path, "*.txt")
        file_list = sorted(glob.glob(file_pattern))  # Sort the file list

        for file_path in file_list:
            with open(file_path, 'r') as file:
                text = file.read()
                outfile.write(text)
                outfile.write("\n")

# negative is done


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()
    return text


def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


positive_reviews = []
negative_reviews = []
with open("combined_text_for_pos_sen.txt", 'r') as file:
    reviews = file.read().split("\n\n")
    positive_reviews = [(preprocess_text(review), 1) for review in reviews if review]

with open("combined_text_for_neg_sen.txt", 'r') as file:
    reviews = file.read().split("\n\n")
    negative_reviews = [(preprocess_text(review), 0) for review in reviews if review]

reviews = positive_reviews + negative_reviews

x = [review[0] for review in reviews]
y = [review[1] for review in reviews]

tfidf = TfidfVectorizer()

x = tfidf.fit_transform(x)

pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names_out())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# LogisticRegression 85.5%
clf = LogisticRegression(random_state=0)
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)
train_acc = accuracy_score(y_train, y_train_pred)
y_pred = clf.predict(x_test)
test_acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy: train =", train_acc * 100, "\ttest =", test_acc * 100)
# print(classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# linear svm  86%
lin_svm = svm.SVC(kernel='linear', random_state=0, C=1.0, probability=True)
lin_svm.fit(x_train, y_train)
train_pred = lin_svm.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
y_pred2 = lin_svm.predict(x_test)
test_acc = accuracy_score(y_test, y_pred2)
print("Linear SVM model accuracy: train =", train_acc * 100, '\ttest= ', test_acc * 100)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred2))

pred_prob = lin_svm.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# naive bayes  81.75%
NB = naive_bayes.MultinomialNB(alpha=0.2)
NB.fit(x_train, y_train)
nb_train = NB.predict(x_train)
train_acc = accuracy_score(y_train, nb_train)
y_nb = NB.predict(x_test)
acc = accuracy_score(y_test, y_nb)
print("Naive Bayes model accuracy: train =", train_acc * 100, '\ttest =', acc * 100)

# visualizing results
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

feature_importances = clf.feature_importances_
sorted_indices = feature_importances.argsort()[::-1]
feature_names = tfidf.get_feature_names_out()
top_N = 20
top_features = [feature_names[i] for i in sorted_indices[:top_N]]  # Assuming you have feature names

plt.figure(figsize=(10, 6))
plt.bar(range(top_N), feature_importances[sorted_indices[:top_N]], align='center')
plt.xticks(range(top_N), top_features, rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top {} Feature Importances'.format(top_N))
plt.tight_layout()
plt.show()

# model saving
pkl_file = "script_NLP.pkl"
with open(pkl_file, 'wb') as file:
    pickle.dump(clean_text, file)
    pickle.dump(preprocess_text, file)
    pickle.dump(tfidf, file)
    pickle.dump(clf, file)
    pickle.dump(lin_svm, file)
    pickle.dump(NB, file)
