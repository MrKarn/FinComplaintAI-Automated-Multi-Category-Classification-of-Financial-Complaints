# FinComplaintAI-Automated-Multi-Category-Classification-of-Financial-Complaints


Objective:
The goal of this project is to classify consumer complaints into predefined categories related to financial products and services. This is a supervised text classification problem where machine learning models are trained on labeled data to predict the category of future complaints based on their content.

Data Source:
The dataset used is the Consumer Complaint Database, a public dataset downloaded from data.gov on May 13, 2019. It contains real-world complaints about financial products and services, with each complaint labeled with a specific product category.

Key Steps:

Data Preprocessing:

Filtered the dataset to include only relevant columns: Product and Consumer complaint narrative.

Removed missing values and renamed columns for simplicity.

Reduced the number of product categories from 18 to 13 by merging similar categories.

Encoded product categories into numerical IDs for model training.

Exploratory Data Analysis (EDA):

Analyzed the distribution of complaints across product categories.

Visualized the most common complaint categories using bar charts.

Text Preprocessing:

Applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.

Removed stop words and used unigrams and bigrams to capture important terms.

Model Training:

Split the data into training (75%) and testing (25%) sets.

Evaluated four machine learning models:

Random Forest

Linear Support Vector Machine (LinearSVC)

Multinomial Naive Bayes

Logistic Regression

Used 5-fold cross-validation to assess model performance.

Model Evaluation:

LinearSVC achieved the highest mean accuracy (77.91%) and was selected as the best model.

Evaluated the model using metrics such as precision, recall, F1-score, and confusion matrix.

Identified misclassified complaints and analyzed the reasons for misclassification.

Predictions:

Tested the model on new, unseen complaints and verified its predictions against the actual labels.

Key Metrics:

Mean Accuracy (Cross-Validation): 77.91% (LinearSVC)

Precision, Recall, F1-Score: Varied by category, with higher scores for categories like "Mortgage" and "Student loan."

Confusion Matrix: Showed clear diagonal dominance, indicating correct classifications, with some misclassifications between similar categories (e.g., "Debt collection" and "Credit reporting").

Results:

The model successfully classified new complaints into the correct categories, demonstrating its ability to generalize to unseen data.

Misclassifications were primarily due to overlapping terms in similar categories (e.g., "Debt collection" and "Credit reporting").

Tools and Libraries Used:

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn (TF-IDF, LinearSVC, Random Forest, Naive Bayes, Logistic Regression)

Jupyter Notebook
