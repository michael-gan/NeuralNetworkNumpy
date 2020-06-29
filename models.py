from activations import *
from initializers import *
from optimizers import *
from losses import *
from metrics import *
from functools import partial
from sklearn.datasets import load_boston, california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tqdm
import time
import gc


class neural_network(object):

    def __init__(self, **kwargs):
        """
        默认使用batch_normalize
        :param kwargs: dict
        """
        allowed_kwargs = ["layer_schema", "input_shape", "learning_rate", "max_iter", "lambda", "activations",
                          "output_layer", "dropout", "combination_method", "decay"]
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        self.learning_rate = None
        self.lamb = None
        self.max_iter = None
        self.batch_size = None
        self.epochs = None
        self.loss_history = []
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
        if "lamb" not in kwargs:
            self.lamb = 1e-2
        else:
            self.lamb = kwargs["lamb"]
        if "combination_method" not in kwargs:
            self.combination_method = linear_combination
        else:
            assert type(kwargs["combination_method"]) == type(linear_combination), "given combination method must be " \
                                                                                   "function, but not %s" % type(kwargs["combination_method"])
            self.combination_method = kwargs["combination_method"]
        if "decay" in kwargs:
            self.decay = kwargs["decay"]
        else:
            self.decay = 1.

    def _neural_network_init(self, input_shape, initilizer):
        if self.weights == {}:
            weight_shape = input_shape
            for i in range(self.num_hidden_layers):
                weight_shape = (self.layer_schema[i], weight_shape[0])
                bias_shape = (weight_shape[0], 1)
                self.weights["layer_%r" % (i + 1)] = initilizer(weight_shape, num_layers=self.num_hidden_layers)
                self.bias["layer_%r" % (i + 1)] = initilizer(bias_shape, num_layers=self.num_hidden_layers)
                # self.bias["layer_%r" % (i + 1)] = np.zeros(bias_shape)
        else:
            pass

    def __temp_var_clean(self, var_name):
        del var_name
        gc.collect()

    def fit(self, data, target, batch_size=None, epochs=None, optimizer=None, initilizer=None, loss=None, verbose=False):
        if np.ndim(data) != 2:
            raise ValueError("given train data should be 2-dim, not %r-dim" % np.ndim(data))
        if batch_size is None:
            self.batch_size = max(64, 2 ** int(np.log2(len(data) / 20)))
        else:
            self.batch_size = batch_size
        num_batches = len(data) // self.batch_size
        if epochs is None:
            self.epochs = 200
        else:
            self.epochs = epochs
        if optimizer is None:
            self.optimizer = Adam
        else:
            self.optimizer = optimizer
        if initilizer is None:
            initilizer = MSRA(seed=1)
        if loss is None:
            self.loss = loss_function.mean_squared_error
        else:
            self.loss = loss
        target = target[np.newaxis, :]
        data = data.T
        self._neural_network_init(data.shape, initilizer)
        new_data = np.concatenate([data, target], axis=0)
        index = np.arange(len(new_data[0]))
        np.random.shuffle(index)
        new_data = new_data[:, index]
        opt_dw = self.optimizer(list(dict(sorted(self.weights.items(), reverse=True)).values()),
                                learning_rate_init=self.learning_rate)
        opt_db = self.optimizer(list(dict(sorted(self.bias.items(), reverse=True)).values()),
                                learning_rate_init=self.learning_rate)
        for i in range(self.epochs):
            for j in range(num_batches + 1):
                batch_data = new_data[:-1, j*self.batch_size:(j+1)*self.batch_size]
                batch_target = new_data[-1:, j*self.batch_size:(j+1)*self.batch_size]
                temp_out = self.__forward(batch_data).reshape(batch_target.shape)
                dw, db = self.__backward(temp_out, batch_target)

                alpha = 1 / (1 + self.decay * i) * self.learning_rate
                # #########################################
                # opt = self.optimizer((self.weights["layer_%r" % i], self.bias["layer_%r" % i]), learning_rate_init=self.learning_rate)
                opt_dw.update_params(dw, alpha=alpha)
                opt_db.update_params(db, alpha=alpha)
                for n in range(self.num_hidden_layers, 0, -1):
                    self.weights["layer_%r" % n] = opt_dw.params[self.num_hidden_layers - n]
                    self.bias["layer_%r" % n] = opt_db.params[self.num_hidden_layers - n]
                # #########################################

            if verbose:
                if (i+1) % 10 == 0:
                    predict = self.__forward(new_data[:-1, :])
                    temp_loss = loss_function.mean_squared_error(new_data[-1, :], predict)[0]
                    self.loss_history.append(temp_loss)
                    print("epoches %d / %r, loss: %r" % (i+1, self.epochs, temp_loss))
        self.__temp_var_clean(self.__temp_z)
        return self

    def predict(self, test_data):
        predict = self.__forward(test_data.T)
        self.__temp_var_clean(self.__temp_z)
        return predict

    def __forward(self, data):
        self.__temp_z = {"layer_0": data}
        for i in range(self.num_hidden_layers):
            data = self.activations["layer_%r" % (i+1)](
                self.combination_method(x=data, w=self.weights["layer_%r" % (i+1)], b=self.bias["layer_%r" % (i+1)])
            )
            self.__temp_z["layer_%r" % (i+1)] = data
        return data

    def __backward(self, predict, target):
        partial_loss = partial(self.loss, y_true=target.T)
        grad_a = gradient(partial_loss, y_pred=predict.T)
        grad_a[abs(grad_a) < loss_function.epsilon] = loss_function.epsilon
        dw = []
        db = []
        for i in range(self.num_hidden_layers, 0, -1):
            temp_d_activation = gradient(self.activations["layer_%r" % i], x=self.__temp_z["layer_%r" % i])
            temp_dz = grad_a * temp_d_activation
            temp_d_combination_dw = self.__temp_z["layer_%r" % (i-1)]
            temp_d_combination_da = self.weights["layer_%r" % i]
            temp_dw = temp_dz.dot(temp_d_combination_dw.T) / target.shape[1]
            temp_db = np.sum(temp_dz, axis=1, keepdims=True) / target.shape[1]
            grad_a = temp_d_combination_da.T.dot(temp_dz)
            dw.append(temp_dw)
            db.append(temp_db)
        return dw, db


class neural_network_regression(neural_network):

    def __init__(self, **kwargs):
        self.__init_para = kwargs
        super().__init__(**kwargs)
        if "output_layer" not in kwargs:
            raise TypeError('Keyword argument must contain output_layer')
        assert hasattr(kwargs["output_layer"], '__name__'), "given output_layer should be method of activation function"
        if kwargs["output_layer"].__name__ not in dir(activations):
            raise TypeError('output_layer should be specified as instance of activation function')
        self.activations["layer_%r" % (self.num_hidden_layers + 1)] = kwargs["output_layer"]

    def __re_init(self):
        self.__init__(**self.__init_para)

    def fit(self, data, target, batch_size=None, epochs=None, optimizer=None, initilizer=None, loss=None, verbose=False):
        if loss is None:
            self.loss = loss_function.mean_squared_error
        else:
            self.loss = loss
        self.__re_init()
        self.num_hidden_layers += 1
        self.layer_schema += (1,)
        super().fit(data, target, batch_size=batch_size, epochs=epochs, optimizer=optimizer, initilizer=initilizer, loss=loss, verbose=verbose)

    # def predict(self, test_data):
    #     pass

    # def __forward(self, input):
    #     print("yes")

    # def __backward(self):
    #     pass


class neural_network_classification(neural_network):

    def __init__(self):
        pass


if __name__ == '__main__':
    model = neural_network_regression(layer_schema=(32, 32, 32), output_layer=activations.relu,
                                      activations=(activations.tanh, activations.tanh, activations.tanh), learning_rate=1.5*1e-3)
    # model = MLPRegressor((16, 16, 16))
    start_time = time.time()
    # train = np.arange(100).reshape(25, 4)
    # target = np.ones((25, 1))
    train, target = california_housing.fetch_california_housing(return_X_y=True)[0], california_housing.fetch_california_housing(return_X_y=True)[1]
    # train, target = load_boston(return_X_y=True)[0], load_boston(return_X_y=True)[1]
    scale = StandardScaler()
    train = scale.fit_transform(train)
    model.fit(train, target, optimizer=Adam, verbose=True, epochs=200)
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(model.predict(train).ravel(), target))
    end_time = time.time()
    print(end_time - start_time)
    # print(model.predict(train))#.shape)
    # print(target)
    plt.figure()
    plt.plot(target)
    plt.plot(model.predict(train).ravel())
    plt.show()
    # print(model.weights)
    # print(model.activations)
    # print(model.num_hidden_layers)
    # print(model.predict(train))
    # print(model.dropout)
