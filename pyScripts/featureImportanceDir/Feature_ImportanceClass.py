import numpy as np

class Build_Feature_Importance:
    def __init__(self, model, X_train, y_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
    
    def fitModel(self):
        self.model.fit(self.X_train, self.y_train)
        print('Model fitted successfully')

    def calculateFeatureImportance(self):  # Corrected method name
        self.feature_importance = self.model.feature_importances_
        self.sorted_idx = np.argsort(self.feature_importance)
        self.feature_names = self.feature_names[self.sorted_idx]
        self.feature_importance = self.feature_importance[self.sorted_idx]
        self.feature_importance = self.feature_importance[-40:]
        self.feature_names = self.feature_names[-40:]
        return self.feature_importance, self.feature_names
    
    def plotFeatureImportance(self, n_features=40,model_name='model',save=False,show=True):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 6))
        plt.barh(range(n_features), self.feature_importance, align='center', color='skyblue')
        plt.yticks(np.arange(n_features), self.feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'{model_name} Feature Importance')
        plt.gca().invert_yaxis()


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

    def SavePlots(self, model_name):
        import matplotlib.pyplot as plt
        self.plotFeatureImportance(model_name=model_name, n_features=40, save=True, show=False)
        plt.savefig(f'featureImportanceDir/{model_name}_feature_importance.png')
        plt.close()
        print(f'Plots for {model_name} saved successfully')