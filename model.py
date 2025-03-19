import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class DiseaseModel:
    """Manages machine learning model selection, training, and evaluation."""

    def __init__(self, model_name='decision_tree', config=None, train_data=None, test_data=None):
        self.config = config
        self.model_name = model_name
        self.train_features, self.train_labels, self.train_df = train_data
        self.test_features, self.test_labels, self.test_df = test_data
        self.model_save_path = self.config.get('model_save_path', './saved_models/')

    def _train_val_split(self):
        """Split dataset into training and validation sets."""
        return train_test_split(
            self.train_features, self.train_labels,
            test_size=self.config['dataset'].get('validation_size', 0.2),
            random_state=self.config.get('random_state', 42)
        )

    def _get_model(self):
        """
        Initializes and returns a machine learning model based on the selected type.
    
        Returns:
            sklearn model: The chosen machine learning classifier.
        """
        if self.model_name == 'mnb':
            return MultinomialNB()

        elif self.model_name == 'decision_tree':
            criterion = self.config['model']['decision_tree']['criterion']
            return DecisionTreeClassifier(criterion=criterion)

        elif self.model_name == 'random_forest':
            n_estimators = self.config['model']['random_forest']['n_estimators']
            return RandomForestClassifier(n_estimators=n_estimators)

        elif self.model_name == 'gradient_boost':
            n_estimators = self.config['model']['gradient_boost']['n_estimators']
            criterion = self.config['model']['gradient_boost']['criterion']
            return GradientBoostingClassifier(n_estimators=n_estimators, criterion=criterion)

        else:
            print(f"Warning: {self.model_name} is not recognized. Defaulting to DecisionTreeClassifier.")
            return DecisionTreeClassifier()


    def train_model(self):
        """Train the ML model and evaluate it."""
        X_train, X_val, y_train, y_val = self._train_val_split()
        model = self._get_model()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        conf_mat = confusion_matrix(y_val, y_pred)
        clf_report = classification_report(y_val, y_pred)
        cv_scores = cross_val_score(model, X_val, y_val, cv=3)

        print(f"\nModel: {self.model_name}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{conf_mat}")
        print(f"Classification Report:\n{clf_report}")
        print(f"Cross-validation Scores: {cv_scores}")

        joblib.dump(model, f"{self.model_save_path}{self.model_name}.joblib")

    def make_prediction(self, saved_model_name=None, test_data=None):
        """Make predictions using the trained model."""
        model_name = saved_model_name or self.model_name
        try:
            model = joblib.load(f"{self.model_save_path}{model_name}.joblib")
        except FileNotFoundError:
            print(f"Model {model_name} not found. Train the model first.")
            return None

        predictions = model.predict(test_data) if test_data is not None else model.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, predictions)
        clf_report = classification_report(self.test_labels, predictions)

        return accuracy, clf_report
