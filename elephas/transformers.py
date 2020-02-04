from pyspark.sql import Row
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

class Transformer(object):
    """Interface which defines a transformer object."""

    def transform(self, dataframe):
        """Transforms the dataframe into an other dataframe.
        # Returns
            The transformed dataframe.
        """
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
        return dataframe.rdd.map(self._transform).toDF()

    def new_dataframe_row(self, old_row, column_name, column_value):
        row = Row(*(old_row.__fields__ + [column_name]))(*(old_row + (column_value,)))

        return row

    def get_index(self, vector):
        """Returns the index with the highest value or with activation threshold."""
        max = 0.0
        max_index = 0
        for index in range(0, self.num_classes):
            if vector[index] > max:
                max = vector[index]
                max_index = index

        return max_index

    def _transform(self, row):
        prediction = row[self.probability_vector_col].toArray() #numpy array
        context = row[self.context_col]
        correct_activity = row[self.correct_activity_col]
        correct_activity = int(correct_activity)
        index = 0.0
        if context is None:
            index = float(self.get_index(prediction))
        else:
            parameters = GatewayParameters(address=self.java_gateway_address, port=self.java_gateway_port, auto_convert=True)
            gateway = JavaGateway(gateway_parameters=parameters)
            entry_point = gateway.entry_point
            index = entry_point.refinePrediction(prediction.tolist(), correct_activity, context)
            index = float(index)
        new_row = self.new_dataframe_row(row, self.output_column, index)

        return new_row



