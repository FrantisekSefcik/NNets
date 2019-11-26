from datetime import datetime
from numpy import expand_dims
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
import tensorflow as tf
import sys
sys.path.append("..")
from src.models.c_dcgan import GAN
from src.models.train_helpers import CGanTrainer

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def load_real_samples():
    (train_X, train_y), (_, _) = load_data()
    x = expand_dims(train_X, axis=-1)
    x = x.astype('float32')
    x = (x - 127.5) / 127.5
    return [x, train_y]


def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    images, labels = images[ix], labels[ix]
    return [images, labels]


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, callbacks, n_epochs=100, n_batch=128):

    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")

    trainer = CGanTrainer(d_model, g_model, gan_model, batch_size=n_batch, n_classes=10, noise_dim=latent_dim)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            [x_real, labels_real] = generate_real_samples(dataset, half_batch)
            real_loss, fake_loss, gan_loss = trainer.train_step(x_real, labels_real, i)

        with file_writer.as_default():
            tf.summary.scalar('loss', data=real_loss, step=i)
            tf.summary.scalar('fake loss', data=fake_loss, step=i)
            tf.summary.scalar('gan loss', data=gan_loss, step=i)

        print('EPOCH:', i)
        trainer.save_fig(i)
    g_model.save('cgan_generator.h5')


latent_dim = 100
gan = GAN((28, 28, 1))
d_model = gan.discriminator
g_model = gan.generator
gan_model = gan.combined
dataset = load_real_samples()

log_path = './log'
callback = tf.keras.callbacks.TensorBoard(log_path)
callback.set_model(gan_model)

# train model
train(g_model, d_model, gan_model, dataset, latent_dim, callback, 50)
