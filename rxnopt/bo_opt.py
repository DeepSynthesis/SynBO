from .bo_test import BayesianOptimizer


class Optimizer:
    def __init__(self, numerical_data, name_data):
        self.numerical_data = numerical_data
        self.name_data = name_data
        # self.done_data = done_data

    def optimize(self, training_X, training_y, method="default", batch_size=5):
        match method:
            case "default":
                bo = BayesianOptimizer()
                bo.optimize(training_X, training_y, self.numerical_data, batch_size=batch_size)
                from IPython import embed

                embed()
                exit()
            case _:
                raise ValueError(f"Method {method} is not supported")
