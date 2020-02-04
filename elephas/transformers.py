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

    def __init__(self, java_gateway_address, java_gateway_port, input_col="prediction", context_col="context", output_col="refined_index"):
        self.input_column = input_col
        self.output_column = output_col
        self.context_col = context_col
        self.java_gateway_address = java_gateway_address
        self.java_gateway_port = java_gateway_port

    def transform(self, dataframe):
        return dataframe.rdd.map(self._transform).toDF()

    def _transform(self, row):
        prediction = row[self.input_column].toArray() #numpy array
        context = row[self.context_col]
        index = 0.0
        if context is None:
            index = float(self.get_index(prediction))
        else:
            parameters = GatewayParameters(address=self.java_gateway_address, port=self.java_gateway_port, auto_convert=True)
            gateway = JavaGateway(gateway_parameters=parameters)
            entry_point = gateway.entry_point
            index = entry_point.refinePrediction(prediction.tolist(), 0, context)
            index = float(index)
        new_row = self.new_dataframe_row(row, self.output_column, index)

        return new_row

    def new_dataframe_row(old_row, column_name, column_value):
        row = Row(*(old_row.__fields__ + [column_name]))(*(old_row + (column_value,)))

        return row

    def get_index(self, vector):
        """Returns the index with the highest value or with activation threshold."""
        max = 0.0
        max_index = self.default_index
        for index in range(0, self.output_dimensionality):
            if vector[index] >= self.activation_threshold:
                return index
            if vector[index] > max:
                max = vector[index]
                max_index = index

        return max_index