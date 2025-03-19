# Diseases_Detection_DS_Project
This project is a machine learning-based disease detection system that can predict diseases based on input symptoms. The project uses multiple machine learning models to analyze data and provide predictions. It is designed to be easy to use, allowing users to train models, test them, and make predictions.

# What This Project Does
1. Loads medical symptom data and prepares it for training.
2. Trains machine learning models to detect diseases.
3. Evaluates the models to check accuracy and reliability.
4. Provides an API for predictions, so users can input symptoms and get disease predictions.
5. Allows easy testing and visualization of model performance.

# Files and Their Purpose

  Diseases_Detection_ML_DS_Project/
  │── README.md               # Project documentation
  │── config.yaml             # Configuration file (stores settings)
  │── requirements.txt        # List of required Python libraries
  │── main.py                 # Runs the full model training process
  │── data_loader.py          # Handles loading and processing datasets
  │── model.py                # Defines machine learning models and trains them
  │── infer.py                # Uses trained models to make predictions
  │── visualization.py        # Generates graphs and charts for data analysis
  │── utils.py                # Contains helper functions for the project
  │── dataset/                # Contains the training and test datasets  
  │   ├── training_data.csv   # The dataset used for training the model
  │   ├── test_data.csv       # The dataset used for testing the model
  │── trained_data/           # Stores trained machine learning models
  │   ├── decision_tree.joblib  # Saved Decision Tree model
  │   ├── random_forest.joblib  # Saved Random Forest model
  │   ├── gradient_boost.joblib # Saved Gradient Boosting model
  │   ├── mnb.joblib            # Saved Naive Bayes model
  │── tests/                  # Unit tests to ensure correctness of code

# How to Set Up and Run the Project

1. **Install Python (if not installed)**
  Make sure you have Python 3.9 or newer installed. You can download it from: 🔗
  https://www.python.org/downloads/

2. **Clone the Repository (Download the Code)**
  git clone https://github.com/yourusername/Diseases_Detection_ML_DS_Project.git
  cd Diseases_Detection_ML_DS_Project

3. **Install Required Dependencies**
   This installs all necessary Python packages, including machine learning libraries.
     pip install -r requirements.txt
   
4. **Train the Model**
   Run the following command to train the model:
      python main.py

5. **Test the Model**
   After training, you can make predictions using the trained model:
     python infer.py --model decision_tree --input sample_input.json

6.  **Running Tests**
    If you want to ensure the project is working correctly, run:
      python -m unittest discover tests




