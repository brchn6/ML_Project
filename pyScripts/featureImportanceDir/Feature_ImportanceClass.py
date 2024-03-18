import numpy as np

class Build_Feature_Importance:
    def __init__(self, model, X_train, y_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
    
    def fitModel(self):
        self.model.fit(self.X_train, self.y_train)
    
    def calculateFeatureImportance(self):  
        self.feature_importance = self.model.feature_importances_
        self.sorted_idx = np.argsort(self.feature_importance).ravel()
        self.feature_names = self.feature_names[self.sorted_idx]
        self.feature_importance = self.feature_importance[self.sorted_idx]
        self.feature_importance = self.feature_importance
        self.feature_names = self.feature_names
        return self.feature_importance, self.feature_names
    
    
    def plotFeatureImportance(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.barh(self.feature_names, self.feature_importance)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance')
        plt.show()

    def plotOriginalFeatureSums(self):
        import matplotlib.pyplot as plt
        sums = self.calculateSumOfOriginalFeatures()
        features = list(sums.keys())
        importance = list(sums.values())

        plt.figure(figsize=(10, 6))
        plt.barh(features, importance, color='skyblue')
        plt.xlabel('Feature Importance Sum')
        plt.title('Sum of Feature Importance for Original Features')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
        plt.show()