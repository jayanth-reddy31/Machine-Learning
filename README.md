# Machine-Learning

**1. Big Mart Sales Prediction**

*   **Project Goal:** This project aims to build a regression model to **predict sales** for Big Mart outlets based on a provided dataset.
*   **Data:** The project uses the `big mart sales.csv` dataset.
    *   It includes categorical features such as `Item_Identifier`, `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type`.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Handling missing values: `Item_Weight` is imputed with the mean, and missing `Outlet_Size` values are replaced with the mode specific to each `Outlet_Type`.
    *   Data analysis and visualization are performed using seaborn and matplotlib, including distribution plots for numerical features (`Item_Weight`, `Item_Visibility`, `Item_MRP`) and count plots for categorical features (`Outlet_Establishment_Year`, `Item_Fat_Content`, `Item_Type`, `Outlet_Size`).
    *   Data preprocessing involves standardizing inconsistent entries in `Item_Fat_Content` (e.g., 'low fat', 'LF', 'Low Fat' all become 'Low Fat') and applying Label Encoding to convert categorical values into numerical format.
    *   The data is split into features (x) and the target variable (y), which is `Item_Outlet_Sales`.
    *   The dataset is further split into training (80%) and testing (20%) sets.
*   **Model:** An **XGBoost Regressor** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **R-squared score** on both the training and test datasets.
*   **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.preprocessing.LabelEncoder`, `sklearn.model_selection.train_test_split`, `xgboost.XGBRegressor`, and `sklearn.metrics` are used.

**2. Breast Cancer Detection**

*   **Project Goal:** This project focuses on building a classification model to **detect breast cancer** (classify a tumor as benign or malignant).
*   **Data:** The project uses a dataset loaded directly from `sklearn.datasets.load_breast_cancer()`.
    *   The target variable, `label`, indicates whether the tumor is malignant (0) or benign (1).
*   **Key Steps:**
    *   Data collection by loading the dataset from scikit-learn and converting it into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, missing values, statistical data, and the distribution of the target variable.
    *   Grouping data by the `label` to observe mean values.
    *   Splitting the data into features (x) and the target (y).
    *   Splitting the dataset into training (80%) and testing (20%) data with `random_state=2`.
*   **Model:** A **Logistic Regression** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict the classification (benign or malignant) for a new input data instance.
*   **Libraries:** `numpy`, `pandas`, `sklearn.datasets`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LogisticRegression`, and `sklearn.metrics.accuracy_score` are used.

**3. Calories Burnt Prediction**

*   **Project Goal:** This project aims to build a regression model to **predict the number of calories burnt** during exercise.
*   **Data:** The project combines data from two CSV files: `calories.csv` and `exercise.csv`. These are loaded into pandas DataFrames and concatenated.
*   **Key Steps:**
    *   Data collection by loading the two datasets and merging them into a single DataFrame.
    *   Basic data exploration, including checking shape, information, and missing values.
    *   Data analysis includes getting statistical data.
    *   Data visualization is performed using seaborn, including count plots for categorical values (`Gender`) and distribution plots for numerical columns (`Age`, `Height`, `Weight`, `Duration`).
    *   Finding the correlation between the data features, visualized using a heatmap.
    *   Converting categorical data (`Gender`) into numerical data (male: 1, female: 0).
    *   Separating features (x) and the target variable (y), which is `Calories`.
    *   Splitting the data into training and testing sets (80% train, 20% test).
*   **Model:** An **XGBoost Regressor** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **R-squared value** and the **Mean Absolute Error** on the test data.
*   **Predictive System:** Demonstrates how to use the trained model to predict the calories burnt for a new input data instance.
*   **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `sklearn.model_selection.train_test_split`, `xgboost.XGBRegressor`, `sklearn.metrics`, and `seaborn` are used.

**4. Car Price Prediction**

*   **Project Goal:** This project aims to build regression models to **predict car prices**.
*   **Data:** The project uses the `car data.csv` dataset.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, missing values, and the distribution of categorical data (`Fuel_Type`, `Seller_Type`, `Transmission`).
    *   Encoding categorical data into numerical values (`Fuel_Type`: Petrol=0, Diesel=1, CNG=2; `Seller_Type`: Dealer=0, Individual=1; `Transmission`: Manual=0, Automatic=1).
    *   Splitting the dataset into features (x), excluding 'Car_Name' and 'Selling_Price', and the target variable (y), which is 'Selling_Price'.
    *   Splitting the data into training (90%) and testing (10%) sets.
*   **Models:** Two regression models are trained:
    *   **Linear Regression**.
    *   **Lasso Regression**.
*   **Evaluation:** Model performance is evaluated using the **R-squared error** on both training and test data. The project also includes visualizations comparing actual and predicted prices using scatter plots.
*   **Libraries:** `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LinearRegression`, `sklearn.linear_model.Lasso`, and `sklearn.metrics` are used.

**5. Credit Card Fraud Detection**

*   **Project Goal:** This project aims to build a classification model to **detect fraudulent credit card transactions**.
*   **Data:** The project uses the `creditcard.csv` dataset.
    *   The dataset is described as highly unbalanced, with 'Class' 0 representing legit transactions and 1 representing fraudulent transactions.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking information, missing values, and the distribution of the 'Class' variable.
    *   Separating data into legit and fraud transactions for analysis.
    *   Comparing statistical data between legit and fraud transactions, particularly for the 'Amount'.
    *   **Undersampling** the majority class (legit transactions) to match the number of fraudulent instances (492), creating a balanced `new_data` DataFrame by concatenating a sample of legit transactions with all fraud transactions.
    *   Splitting the balanced data into features (x) and the target variable (y), which is 'Class'.
    *   Splitting the data into training (80%) and testing (20%) sets, ensuring stratification based on the target variable and using `random_state=2`.
*   **Model:** A **Logistic Regression** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict whether a new transaction is legit or fraud.
*   **Libraries:** `pandas`, `numpy`, `matplotlib.pyplot`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LogisticRegression`, and `sklearn.metrics.accuracy_score` are used.

**6. Customer Segmentation using K-Means Clustering**

*   **Project Goal:** This project aims to perform **customer segmentation** using clustering to group customers based on their features.
*   **Data:** The project uses the `Mall_Customers.csv` dataset.
    *   Specifically, it focuses on the 'Annual Income' and 'Spending Score' columns as features for clustering.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, and null values.
    *   Selecting the relevant features ('Annual Income' and 'Spending Score') and extracting their values into a numpy array `x`.
    *   Determining the **optimal number of clusters** (k) using the **Elbow Method**. This involves calculating the Within-Cluster Sum of Squares (WCSS) for a range of k values and plotting the results to find the "elbow point" where the decrease in WCSS becomes marginal.
    *   Training a **K-Means clustering model** with the determined optimal number of clusters (which is identified as 5 based on the Elbow Method).
    *   Predicting the cluster label for each data point.
    *   Visualizing the clusters and their centroids using a scatter plot.
*   **Model:** **K-Means Clustering** is used for unsupervised learning.
*   **Evaluation:** The **Elbow Method** using **WCSS** is the primary technique used to evaluate and select the appropriate number of clusters, rather than a traditional supervised learning metric.
*   **Libraries:** `numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, and `sklearn.cluster.KMeans` are used.

**7. Fake News Prediction**

*   **Project Goal:** This project aims to build a classification model to **predict whether a news article is fake or real**.
*   **Data:** The project uses the `train.csv` dataset.
    *   Key columns include `id`, `title`, `author`, `text`, and `label`, where the label indicates fake (1) or real (0) news.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape and missing values.
    *   Handling missing values by replacing them with empty strings.
    *   Merging the `author` and `title` columns into a new `content` column for processing.
    *   Applying **Stemming** using `PorterStemmer` to reduce words to their root form. This process also involves removing special characters and numbers, converting text to lowercase, splitting text into words, and removing English stopwords.
    *   Separating the processed `content` (features) and the `label` (target).
    *   Converting textual data into numerical data using **TfidfVectorizer**.
    *   Splitting the data into training (80%) and testing (20%) sets, ensuring stratification based on the target variable and using `random_state=2`.
*   **Model:** A **Logistic Regression** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict if a news article is real or fake based on input data.
*   **Libraries:** `numpy`, `pandas`, `re`, `nltk.corpus.stopwords`, `nltk.stem.porter.PorterStemmer`, `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LogisticRegression`, and `sklearn.metrics.accuracy_score` are used. `nltk` is also used to download stopwords.

**8. Gold Price Prediction**

*   **Project Goal:** This project aims to build a regression model to **predict gold prices**.
*   **Data:** The project uses the `gold_price_data.csv` dataset.
    *   The `GLD` column represents the gold price.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, missing values, and statistical measures.
    *   Converting the 'Date' column to datetime objects.
    *   Analyzing the correlation between features, visualized using a heatmap. It specifically checks the correlation with the `GLD` price.
    *   Visualizing the distribution of the `GLD` data using a histogram.
    *   Splitting the data into features (x), excluding 'GLD' and 'Date', and the target variable (y), which is 'GLD'.
    *   Splitting the data into training (80%) and testing (20%) sets with `random_state=2`.
*   **Model:** A **Random Forest Regressor** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **R-squared value** on the test data. A plot comparing actual vs. predicted values is also generated.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict the gold price for new input data.
*   **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection.train_test_split`, `sklearn.ensemble.RandomForestRegressor`, and `sklearn.metrics` are used.

**9. Heart Disease Prediction**

*   **Project Goal:** This project aims to build a classification model to **predict whether a person has heart disease**.
*   **Data:** The project uses the `heart_disease_data.csv` dataset.
    *   The target column 'target' indicates defective heart (1) or normal heart (0).
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, missing values, statistical measures, and the distribution of the target column.
    *   Splitting the data into features (x), excluding 'target', and the target variable (y), which is 'target'.
    *   Splitting the dataset into training (80%) and testing (20%) data, ensuring stratification based on the target and using `random_state=2`.
*   **Model:** A **Logistic Regression** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict whether a person has heart disease based on new input data.
*   **Libraries:** `numpy`, `pandas`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LogisticRegression`, and `sklearn.metrics.accuracy_score` are used.

**10. House Price Prediction**

*   **Project Goal:** This project aims to build a regression model to **predict house prices**.
*   **Data:** The project uses the California Housing dataset, loaded from `sklearn.datasets.fetch_california_housing()`.
    *   The target variable is added to the DataFrame as the 'price' column.
*   **Key Steps:**
    *   Data collection by loading the dataset from scikit-learn and converting it into a pandas DataFrame.
    *   Basic data exploration, including checking shape, missing values, and statistical insights.
    *   Understanding the correlation between various features and the target, visualized using a heatmap.
    *   Splitting the data into features (x), excluding 'price', and the target variable (y), which is 'price'.
    *   Splitting the data into training (80%) and testing (20%) sets with `random_state=1`.
*   **Model:** An **XGBoost Regressor** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **R-squared error** and **Mean Absolute Error** on both the training and test datasets. The actual and predicted prices are also visualized using a scatter plot.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict the house price for new input data.
*   **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection.train_test_split`, `sklearn.datasets`, `xgboost.XGBRegressor`, and `sklearn.metrics` are used.

**11. Loan Prediction**

*   **Project Goal:** This project aims to build a classification model to **predict whether a loan will be approved**.
*   **Data:** The project uses the `train_loan.csv` dataset.
    *   The target variable is 'Loan_Status'.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking type, shape, statistical measures, and the count of missing values.
    *   Handling missing values by **dropping rows** with any missing values.
    *   **Label Encoding** is applied to convert string values into numerical values.
    *   Specific value replacement is done for the 'Dependents' column, changing '3+' to '4'.
    *   Data visualization is performed using count plots to see the relationship between categorical features (Education, Married, Gender) and 'Loan_Status'.
    *   Separating the data into features (x), excluding 'Loan_Status' and 'Loan_ID', and the target variable (y), which is 'Loan_Status'.
    *   Splitting the data into training (90%) and testing (10%) sets, ensuring stratification based on the target and using `random_state=2`.
*   **Model:** An **SVM (Support Vector Machine) Classifier** with a 'linear' kernel is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict whether a loan is approved or not based on new input data.
*   **Libraries:** `numpy`, `pandas`, `seaborn`, `sklearn.model_selection.train_test_split`, `sklearn.svm`, and `sklearn.metrics.accuracy_score` are used.

**12. Medical Insurance Cost Prediction**

*   **Project Goal:** This project aims to build a regression model to **predict medical insurance costs**.
*   **Data:** The project uses the `insurance.csv` dataset.
    *   The target variable is 'charges'.
    *   Categorical values include `sex`, `smoker`, and `region`.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, and missing values.
    *   Data analysis includes getting statistical measures.
    *   Data visualization is performed using seaborn and matplotlib, including distribution plots for numerical features (`age`, `bmi`, `charges`) and count plots for categorical features (`sex`, `children`, `smoker`, `region`).
    *   **Encoding categorical data** into numerical data (`sex`: male=0, female=1; `smoker`: yes=0, no=1; `region`: southwest=0, southeast=1, northwest=2, northeast=3).
    *   Splitting the data into features (x), excluding 'charges', and the target variable (y), which is 'charges'.
    *   Splitting the data into training (80%) and testing (20%) sets with `random_state=2`.
*   **Model:** A **Linear Regression** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **R-squared value** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict the medical insurance cost for new input data.
*   **Libraries:** `numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LinearRegression`, and `sklearn.metrics` are used.

**13. Movie Recommendation System using Cosine Similarity**

*   **Project Goal:** This project aims to build a **content-based movie recommendation system** using the cosine similarity algorithm.
*   **Data:** The project uses the `movies.csv` dataset.
    *   Relevant features selected for recommendation include `genres`, `keywords`, `tagline`, `cast`, and `director`.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Checking the number of rows and columns.
    *   Selecting relevant features.
    *   Handling null values in selected features by replacing them with empty strings.
    *   Combining the selected features into a single string for each movie.
    *   Converting the text data into numerical feature vectors using **TfidfVectorizer**.
    *   Calculating the **cosine similarity score** between all pairs of movies based on their feature vectors.
    *   Taking a user's favorite movie name as input.
    *   Using the `difflib` library to find the closest match to the user's input movie name from the dataset's movie titles.
    *   Getting the index of the closest matching movie.
    *   Getting a list of similarity scores for all movies relative to the input movie.
    *   Sorting the movies based on their similarity score in descending order.
    *   Printing the titles of the top similar movies as recommendations.
*   **Model/Algorithm:** **Cosine Similarity** is the core algorithm used to find similarities between movies based on their content features. **TfidfVectorizer** is used for feature extraction.
*   **Evaluation:** Not explicitly described in terms of standard metrics, but the system's performance is inherently linked to the quality of the similarity scores and resulting recommendations.
*   **Libraries:** `numpy`, `pandas`, `difflib`, `sklearn.feature_extraction.text.TfidfVectorizer`, and `sklearn.metrics.pairwise.cosine_similarity` are used.

**14. Titanic Survival Prediction**

*   **Project Goal:** This project aims to build a classification model to **predict the survival of passengers on the Titanic**.
*   **Data:** The project uses the `titanic data.csv` dataset.
    *   The target variable is 'Survived' (1 for survived, 0 for not survived).
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Basic data exploration, including checking shape, information, and missing values.
    *   Handling missing values: The 'Cabin' column is dropped. Missing values in the 'Age' column are replaced with the mean age. Missing values in the 'Embarked' column are replaced with the mode.
    *   Basic statistical measures are obtained.
    *   The number of people who survived and did not survive is counted.
    *   Data visualization is performed using count plots to see the distribution of 'Survived', 'Sex', and 'Pclass', and how 'Survived' relates to 'Sex' and 'Pclass'.
    *   **Encoding categorical columns** into numerical values (`Sex`: male=0, female=1; `Embarked`: S=0, C=1, Q=2).
    *   Separating the data into features (x), excluding 'PassengerId', 'Ticket', 'Name', and 'Survived', and the target variable (y), which is 'Survived'.
    *   Splitting the data into training (80%) and testing (20%) sets with `random_state=2`.
*   **Model:** A **Logistic Regression** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict the survival outcome for a new passenger based on input data.
*   **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection.train_test_split`, `sklearn.linear_model.LogisticRegression`, and `sklearn.metrics.accuracy_score` are used.

**15. Wine Quality Prediction**

*   **Project Goal:** This project aims to build a classification model to **predict the quality of red wine**.
*   **Data:** The project uses the `winequality-red.csv` dataset.
    *   The target variable is 'quality'.
*   **Key Steps:**
    *   Data collection and loading into a pandas DataFrame.
    *   Checking the number of rows and columns and printing a sample.
    *   Checking for missing values.
    *   Data analysis includes getting statistical measures.
    *   Data visualization is performed, including a categorical plot showing the number of values for each quality level and bar plots showing the relationship between 'volatile acidity' vs. 'quality' and 'citric acid' vs. 'quality'.
    *   Analyzing the correlation between features and the target, visualized using a heatmap.
    *   Separating the data into features (x), excluding 'quality'.
    *   **Label Binarization** is applied to the 'quality' target variable (y), where wines with quality >= 7 are labeled as 1 (good quality) and others as 0 (bad quality).
    *   Splitting the dataset into training (80%) and testing (20%) sets with `random_state=3`.
*   **Model:** A **Random Forest Classifier** model is trained.
*   **Evaluation:** The model's performance is evaluated using the **accuracy score** on both the training and test datasets.
*   **Predictive System:** Includes a demonstration of how to use the trained model to predict the quality (good or bad) for a new wine instance based on input data.
*   **Libraries:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection.train_test_split`, `sklearn.ensemble.RandomForestClassifier`, and `sklearn.metrics.accuracy_score` are used.
