from keras import backend as K
from keras.models import model_from_json
from keras import backend as K
import numpy as np
from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row

class Predictor(object):

    def __init__(self, keras_model):
        self.model = self.serialize_keras_model(keras_model)

    def serialize_keras_model(self,model):
        dictionary = {}
        dictionary['model'] = model.to_json()
        dictionary['weights'] = model.get_weights()

        return dictionary

    def deserialize_keras_model(self,dictionary):
        architecture = dictionary['model']
        weights = dictionary['weights']
        model = model_from_json(architecture)
        model.set_weights(weights)

        return model

    def new_dataframe_row(self, old_row, column_name, column_value):
        row = Row(*(old_row.__fields__ + [column_name]))(*(old_row + (column_value,)))

        return row

    def predict(self, dataframe):
        raise NotImplementedError


class ModelPredictor(Predictor):

    def __init__(self, keras_model, features_col="features", output_col="prediction"):
        super(ModelPredictor, self).__init__(keras_model)
        assert isinstance(features_col, (str, list)), "'features_col' must be a string or a list of strings"
        self.features_column = [features_col] if isinstance(features_col, str) else features_col
        self.output_column = output_col

    def _predict(self, iterator):
        model = self.deserialize_keras_model(self.model)
        for row in iterator:
            features = [np.asarray([row[c]]) for c in self.features_column]
            prediction = model.predict(features)
            dense_prediction = DenseVector(prediction[0])
            new_row = self.new_dataframe_row(row, self.output_column, dense_prediction)
            yield new_row

    def predict(self, dataframe):
        return dataframe.rdd.mapPartitions(self._predict).toDF()