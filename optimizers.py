import numpy as np


class BaseOptimizer:

    def __init__(self, params, learning_rate_init=0.1):
        self.params = [param for param in params]
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, grads, **kwargs):
        if "alpha" in kwargs:
            self.learning_rate = kwargs["alpha"]
        updates = self._get_update(grads)
        params = []
        for param, update in zip(self.params, updates):
            param += update
            params.append(param)
        self.params = params

    def iteration_ends(self, time_step):
        pass

    def trigger_stopping(self, msg, verbose):
        if verbose:
            print(msg + "stopping")
        return True


class SGD(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.1, lr_schedule="constant", momentum=0.9, nesterov=True, power_t=0.5):
        super().__init__(params, learning_rate_init)
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = [np.zeros_like(param) for param in params]

    def iteration_ends(self, time_step):
        if self.lr_schedule == 'invscaling':
            self.learning_rate = (float(self.learning_rate_init)/(time_step + 1) ** self.power_t)

    def trigger_stopping(self, msg, verbose):
        if self.lr_schedule != 'adaptive':
            if verbose:
                print(msg + " stopping.")
            return True

        if self.learning_rate <= 1e-6:
            if verbose:
                print(msg + " Learning rate too small. Stopping.")
            return True

        self.learning_rate /= 5.
        if verbose:
            print(msg + " Setting learning rate to %f" % self.learning_rate)
        return False

    def _get_update(self, grads):
        updates = [self.momentum * velocity - self.learning_rate * grad for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates
        if self.nesterov:
            updates = [self.momentum * velocity - self.learning_rate * grad for velocity, grad in zip(self.velocities, grads)]
        return updates


class Adam(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_update(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2) for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init * np.sqrt(1 - self.beta_2 ** self.t) / (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon) for m, v in zip(self.ms, self.vs)]
        return updates


class Momentum(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.001, gamma=0.9):
        super().__init__(params, learning_rate_init)
        self.gamma = gamma
        self.velocities = [np.zeros_like(param) for param in params]

    def _get_update(self, grads):
        v = [self.gamma * v + self.learning_rate * grad for v, grad in zip(self.velocities, grads)]
        updates = [-update for update in v]
        return updates


class NAG(BaseOptimizer):

    def __init__(self):
        pass

    def _get_update(self, grads):
        pass


class Adagrad(BaseOptimizer):

    def __init__(self):
        pass

    def _get_update(self, grads):
        pass


class Adadelta(BaseOptimizer):

    def __init__(self):
        pass

    def _get_update(self, grads):
        pass


class AdaMax(BaseOptimizer):

    def __init__(self):
        pass

    def _get_update(self, grads):
        pass


class Nadam(BaseOptimizer):

    def __init__(self):
        pass

    def _get_update(self, grads):
        pass


class RMSProp(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(params, learning_rate_init)
        self.epsilon = epsilon
        self.s = [np.zeros_like(param) for param in params]
        self.beta = beta

    def _get_update(self, grads):
        self.s = [self.beta * s + (1 - self.beta) * np.square(grad) for s, grad in zip(self.s, grads)]
        update = [-self.learning_rate * grad / np.sqrt(self.epsilon + s) for s, grad in zip(self.s, grads)]
        return update


