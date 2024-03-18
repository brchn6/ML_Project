import numpy as np

class Build_Feature_Importance:
    def __init__(self, model, X_train, y_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
    
    def fitModel(self):
        self.model.fit(self.X_train, self.y_train)
    
    def calculateFeatureImportance(self):  # Corrected method name
        self.feature_importance = self.model.feature_importances_
        self.sorted_idx = np.argsort(self.feature_importance)
        self.feature_names = self.feature_names[self.sorted_idx]
        self.feature_importance = self.feature_importance[self.sorted_idx]
        self.feature_importance = self.feature_importance[-20:]
        self.feature_names = self.feature_names[-20:]
        return self.feature_importance, self.feature_names
    
    def plotFeatureImportance(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.barh(self.feature_names, self.feature_importance)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance')
        plt.show()

    #build a method to clac the sum of the feature that originated from the same column before the one hot encoding 
    def calculateSumOfOriginalFeatures(self):
        original_feature_names = [name.split('_')[2] if '__' in name else name.split('_')[1] for name in self.feature_names]
        unique_original_features = set(original_feature_names)
        feature_sums = {}
        for feature in unique_original_features:
            indices = [i for i, name in enumerate(original_feature_names) if name == feature]
            feature_sum = np.sum(self.feature_importance[indices])
            feature_sums[feature] = feature_sum
        return feature_sums

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