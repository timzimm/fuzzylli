import logging
from collections import namedtuple

from jax import jit, value_and_grad
import jax.numpy as np
import jax.random as random
import jax.nn as nn
from optax import l2_loss, adam, apply_updates


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mlp_params = namedtuple("mlp_params", ["weights", "biases"])


def init_mlp_params(layer_widths):
    """
    Initializes random multilayer perceptron of topology layer_widths
    """
    weights = []
    biases = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        weights.append(
            random.normal(shape=(n_in, n_out), key=random.PRNGKey(0))
            * np.sqrt(2 / n_in)
        )
        biases.append(np.ones(shape=(n_out,)))

    return mlp_params(weights=weights, biases=biases)


def evaluate_mlp(x, mlp_params):
    """
    Evaluates multilayer perceptron defined by params
    """
    (*hidden_w, last_w), (*hidden_b, last_b) = mlp_params.weights, mlp_params.biases
    x = np.atleast_1d(x)
    for w, b in zip(hidden_w, hidden_b):
        # x = nn.relu(x @ w + b)
        x = nn.tanh(x @ w + b)
    x = x @ last_w + last_b
    return x.squeeze()


def mlp_optimization(epochs, init_params, forward_pass, X_train, y_train):
    """
    Adam-based gradient descent for the mlp parameters enetring the
    forward pass.
    """

    def loss(params, X, y, forward_pass):
        """
        Canonical L2 loss over entire batch
        """
        y_hat = forward_pass(X, params)
        loss_value = l2_loss(y_hat, y).sum(axis=-1)
        return loss_value.mean()

    def fit(params, optimizer, forward_pass, X_train, y_train):
        opt_state = optimizer.init(params)

        @jit
        def step(params, opt_state, X_train, y_train):
            loss_value, grads = value_and_grad(loss)(
                params, X_train, y_train, forward_pass
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = apply_updates(params, updates)
            return params, opt_state, loss_value

        params, opt_state, loss_value = step(params, opt_state, X_train, y_train)
        logger.info(f"Initial loss: {loss_value}")
        for epoch in range(1, epochs):
            params, opt_state, loss_value = step(params, opt_state, X_train, y_train)
        logger.info(f"Final loss: {loss_value}")

        return params

    # Crued but apparantly sufficient
    optimizer = adam(learning_rate=5e-3)

    # fit the rho from margenalisation to the target
    params_logrho_in_df = fit(
        init_params,
        optimizer,
        forward_pass,
        X_train,
        y_train,
    )

    return params_logrho_in_df
