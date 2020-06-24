from activations import *
from initializers import *
from losses import *
from functools import partial
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tqdm
import time


class neural_network(object):

    def __init__(self, **kwargs):
        """
        默认使用batch_normalize
        :param kwargs: dict
        """
        allowed_kwargs = ["layer_schema", "input_shape", "learning_rate", "max_iter", "lambda", "activations",
                          "output_layer", "dropout", "combination_method"]
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        self.learning_rate = None
        self.lamb = None
        self.max_iter = None
        self.batch_size = None
        self.epochs = None
        self.model_sequence = []
        self.weights = {}
        self.bias = {}
        self.activations = {}
        self.dropout = {}
        self.optimizer = None
        self.loss = None
        if "layer_schema" not in kwargs:
            raise NotImplementedError("schema of neural network should be given with kwarg \'layer_schema\'")
        else:
            self.layer_schema = kwargs["layer_schema"]
            self.num_hidden_layers = len(self.layer_schema)
            if "activations" not in kwargs:
                self.activations = dict(
                    [("layer_%r" % (i + 1), activations.tanh) for i in range(self.num_hidden_layers)])
            else:
                self.activations = dict(
                    [("layer_%r" % (i + 1), kwargs["activations"][i]) for i in range(self.num_hidden_layers)])
            if not isinstance(kwargs["layer_schema"], tuple):
                raise TypeError("type of neural network schema should be tuple, but not %s",
                                type(kwargs["layer_schema"]))
        if "dropout" in kwargs:
            if not isinstance(kwargs["dropout"], tuple):
                raise TypeError("dropout layer schema should be tuple, but not %s",
                                type(kwargs["layer_schema"]))
            else:
                if np.any(list(map(lambda x: x < 0 or x > 1, kwargs["dropout"]))):
                    raise ValueError("dropout rate should be in range (0,1)")
                self.dropout = dict(
                    [("layer_%r" % (i + 1), kwargs["dropout"][i]) for i in range(self.num_hidden_layers)])
        else:
            self.dropout = dict([("layer_%r" % (i + 1), 0) for i in range(self.num_hidden_layers)])
        if "learning_rate" not in kwargs:
            self.learning_rate = 1e-2
        else:
            self.learning_rate = kwargs["learning_rate"]
        if "combination_method" not in kwargs:
            self.combination_method = linear_combination
        else:
            assert type(kwargs["combination_method"]) == type(linear_combination), "given combination method must be " \
                                                                                   "function, but not %s" % type(kwargs["combination_method"])
            self.combination_method = kwargs["combination_method"]

    def _neural_network_init(self, input_shape, initilizer):
        if self.weights == {}:
            weight_shape = input_shape
            for i in range(self.num_hidden_layers):
                weight_shape = (self.layer_schema[i], weight_shape[0])
                bias_shape = (weight_shape[0], 1)
                self.weights["layer_%r" % (i + 1)] = initilizer(weight_shape)
                self.bias["layer_%r" % (i + 1)] = initilizer(bias_shape)
        else:
            pass

    def fit(self, data, target, batch_size=None, epochs=None, optimizer=None, initilizer=None, loss=None):
        if np.ndim(data) != 2:
            raise ValueError("given train data should be 2-dim, not %r-dim" % np.ndim(data))
        if batch_size is None:
            self.batch_size = max(32, 2 ** int(np.log2(len(data) / 20)))
        else:
            self.batch_size = batch_size
        if epochs is None:
            self.epochs = 20
        else:
            self.epochs = epochs
        if optimizer is None:
            self.optimizer = "SGD"
        else:
            self.optimizer = optimizer
        if initilizer is None:
            initilizer = RandomNormal()
        if loss is None:
            self.loss = loss_function.mean_squared_error
        else:
            self.loss = loss
        target = target[:, np.newaxis]
        self._neural_network_init(data.T.shape, initilizer)
        for i in tqdm.tqdm(range(self.epochs)):
            temp_out = self._forward(data).reshape(target.shape)
            self._backward(temp_out, target)
        return self

    def predict(self, test_data):
        return self._forward(test_data)

    def _forward(self, data):
        data = data.T
        self.__temp_z = {"layer_0": data}
        for i in range(self.num_hidden_layers):
            data = self.activations["layer_%r" % (i+1)](
                self.combination_method(x=data, w=self.weights["layer_%r" % (i+1)], b=self.bias["layer_%r" % (i+1)])
            )
            self.__temp_z["layer_%r" % (i+1)] = data
        return data

    def _backward(self, predict, target):
        partial_loss = partial(self.loss, y_true=target)
        grad_a = gradient(partial_loss, y_pred=predict)
        for i in range(self.num_hidden_layers, 0, -1):
            temp_d_activation = gradient(self.activations["layer_%r" % i], x=self.__temp_z["layer_%r" % i])
            temp_dz = grad_a * temp_d_activation
            # print("temp_dz shape", temp_dz.shape)
            # partial_w = partial(self.combination_method, x=self.__temp_z["layer_%r" % i], b=self.bias["layer_%r" % i])
            # partial_b = partial(self.combination_method, x=self.__temp_z["layer_%r" % i], w=self.weights["layer_%r" % i])
            # partial_a = partial(self.combination_method, w=self.weights["layer_%r" % i], b=self.bias["layer_%r" % i])
            temp_d_combination_dw = self.__temp_z["layer_%r" % (i-1)]
            # temp_d_combination_db = np.ones_like(self.bias["layer_%r" % i])
            temp_d_combination_da = self.weights["layer_%r" % i]
            # print(temp_d_combination_dw.shape, temp_d_combination_db.shape, temp_d_combination_da.shape)
            temp_dw = temp_dz.dot(temp_d_combination_dw.T)
            # print("temp_dw shape", temp_dw.shape)
            # print(temp_dw.shape)
            temp_db = np.sum(temp_dz, axis=1, keepdims=True)
            # print("temp_db shape", temp_db.shape)
            # grad_a = gradient(partial_a, x=self.__temp_z["layer_%r" % (i - 1)])
            self.weights["layer_%r" % i] -= temp_dw * self.learning_rate
            self.bias["layer_%r" % i] -= temp_db * self.learning_rate
            grad_a = temp_d_combination_da.T.dot(temp_dz)
            # print(grad_a.shape, self.__temp_z["layer_%r" % (i-1)].shape)


class neural_network_regression(neural_network):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "output_layer" not in kwargs:
            raise TypeError('Keyword argument must contain output_layer')
        assert hasattr(kwargs["output_layer"], '__name__'), "given output_layer should be method of activation function"
        if kwargs["output_layer"].__name__ not in dir(activations):
            raise TypeError('output_layer should be specified as instance of activation function')
        self.activations["layer_%r" % (self.num_hidden_layers + 1)] = kwargs["output_layer"]

    def fit(self, data, target, batch_size=None, epochs=None, optimizer=None, initilizer=None, loss=None):
        if loss is None:
            self.loss = loss_function.mean_absolute_error
        else:
            self.loss = loss
        self.num_hidden_layers += 1
        self.layer_schema += (1,)
        super().fit(data, target, batch_size=None, epochs=None, optimizer=None, initilizer=None, loss=loss)

    # def predict(self, test_data):
    #     pass

    # def _forward(self, input):
    #     print("yes")

    # def _backward(self):
    #     pass


class neural_network_classification(neural_network):

    def __init__(self):
        pass


if __name__ == '__main__':
    model = neural_network_regression(layer_schema=(16, 16), output_layer=activations.relu, activations=(activations.tanh, activations.tanh), learning_rate=0.0009)
    model = MLPRegressor()
    # train = np.arange(100).reshape(25, 4)
    # target = np.ones((25, 1))
    train, target = load_boston(return_X_y=True)[0], load_boston(return_X_y=True)[1]
    scale = StandardScaler()
    train = scale.fit_transform(train)
    model.fit(train, target, loss=loss_function.mean_squared_error, epochs=20)
    print(model.predict(train))#.shape)
    print(target)
    plt.figure()
    plt.plot(target)
    plt.plot(model.predict(train).ravel())
    plt.show()
    # print(model.weights)
    # print(model.activations)
    # print(model.num_hidden_layers)
    # print(model.predict(train))
    # print(model.dropout)
