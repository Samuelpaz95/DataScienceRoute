import tensorflow as tf
from keras.losses import Loss
from keras.optimizers import Optimizer
from keras.models import Model


class Trainer:
    def __init__(self, loss: Loss, optimizer: Optimizer) -> None:
        self.loss = loss
        self.optimizer = optimizer

    @tf.function
    def train_step(self, model: Model, x: tf.Tensor, y: tf.Tensor, accuracy=False) -> float:
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = self.loss(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        if accuracy:
            return loss.numpy(), tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1)), tf.float32)).numpy()
        return loss.numpy()

    @tf.function
    def fit(self, model: Model, x: tf.Tensor, y: tf.Tensor, epochs: int, accuracy=False) -> None:
        for epoch in range(epochs):
            loss = self.train_step(model, x, y, accuracy=accuracy)
            print(f'Epoch {epoch + 1} loss: {loss}')
