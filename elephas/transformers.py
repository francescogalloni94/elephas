from pyspark.sql import Row
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

class Transformer(object):

    def new_dataframe_row(self, old_row, column_name, column_value):
        row = Row(*(old_row.__fields__ + [column_name]))(*(old_row + (column_value,)))

        return row

    def transform(self, dataframe):
        raise NotImplementedError

class OntologyTransformer(Transformer):

    def __init__(self, java_gateway_address, java_gateway_port, num_classes, probability_vector_col="prediction",
                 correct_activity_col="label_index", context_col="context", output_col="refined_index"):
        self.probability_vector_col = probability_vector_col
        self.correct_activity_col = correct_activity_col
        self.output_column = output_col
        self.context_col = context_col
        self.java_gateway_address = java_gateway_address
        self.java_gateway_port = java_gateway_port
        self.num_classes = num_classes

    def transform(self, dataframe):
        return dataframe.rdd.mapPartitions(self.partition_transform).toDF()

    def partition_transform(self,iterator):
        parameters = GatewayParameters(address=self.java_gateway_address, port=self.java_gateway_port,
                                       auto_convert=True)
        gateway = JavaGateway(gateway_parameters=parameters)
        entry_point = gateway.entry_point
        new_iterator = []
        for item in iterator:
            new_item = self._transform(item, entry_point)
            new_iterator.append(new_item)
        gateway.close()
        new_iterator = iter(new_iterator)
        return new_iterator


    def get_index(self, vector):
        max = 0.0
        max_index = 0
        for index in range(0, self.num_classes):
            if vector[index] > max:
                max = vector[index]
                max_index = index

        return max_index

    def _transform(self, row, entry_point):
        prediction = row[self.probability_vector_col].toArray() #numpy array
        context = row[self.context_col]
        correct_activity = row[self.correct_activity_col]
        correct_activity = int(correct_activity)
        index = 0.0
        if context is None:
            index = float(self.get_index(prediction))
        else:
            index = entry_point.refinePrediction(prediction.tolist(), correct_activity, context)
            index = float(index)
        new_row = self.new_dataframe_row(row, self.output_column, index)

        return new_row


class LabelIndexTransformer(Transformer):

    def __init__(self, output_dim, input_col="prediction", output_col="prediction_index",
                 default_index=0, activation_threshold=0.55):
        self.input_column = input_col
        self.output_column = output_col
        self.output_dimensionality = output_dim
        self.activation_threshold = activation_threshold
        self.default_index = default_index

    def get_index(self, vector):
        max = 0.0
        max_index = self.default_index
        for index in range(0, self.output_dimensionality):
            if vector[index] >= self.activation_threshold:
                return index
            if vector[index] > max:
                max = vector[index]
                max_index = index

        return max_index

    def _transform(self, row):
        prediction = row[self.input_column]
        index = float(self.get_index(prediction))
        new_row = self.new_dataframe_row(row, self.output_column, index)

        return new_row

    def transform(self, dataframe):
        return dataframe.rdd.map(self._transform).toDF()

