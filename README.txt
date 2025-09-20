========================================================
      FAKE PRODUCT REVIEW DETECTION USING MACHINE LEARNING
========================================================

📁 Project Files:
-----------------
1. python fake_review_detector.py     --> Python script to train and test the model
2. large_dataset.csv               --> Dataset of product reviews (fake & genuine)
3. README.txt                      --> Project documentation (this file)

📌 Requirements:
---------------
Before running the project, install the following Python packages:

> pip install pandas scikit-learn

🧠 Project Description:
-----------------------
This project detects fake product reviews using machine learning. It uses text classification techniques to analyze review content and predict whether a review is fake or genuine.

The model is trained on a labeled dataset using TF-IDF vectorization and Logistic Regression.

🛠️ How to Run:
--------------
1. Open a terminal or command prompt.
2. Navigate to the folder containing the files.
3. Run the following command:

> python fake_review_detector.py

4. The program will:
   - Train the model
   - Evaluate it on a test set
   - Let you enter your own reviews for live prediction

🔎 Example Review Prediction:
-----------------------------
> Enter a review to check (or type 'exit' to quit):
> "Amazing product! Love it so much."

Prediction: Genuine Review ✅

> "Totally fake, not as advertised. Do not buy."

Prediction: Fake Review ❌

📬 Contact:
----------
Created by: [K.Keerthi priya]
For academic/demo use only.
