#This class is used to give any transformer a condition to be applied or not
#Will be used in the GANS preprocessing pipeline to make GANS conditional:
#If the condition is true, the GANS will be applied, otherwise, it will not be applied
#This class is used in the following functions:

from sklearn.base import TransformerMixin

class ConditionalTransformer(TransformerMixin):
    def __init__(self, condition, transformer):
        self.condition = condition
        self.transformer = transformer
    
    def fit(self, X, y=None):
        return self.transformer.fit(X, y)
    
    def transform(self, X):
        if self.condition:
            return self.transformer.transform(X)
        else:
            return X

    def fit_transform(self, X, y=None):
        if self.condition:
            return self.transformer.fit_transform(X, y)
        else:
            return X