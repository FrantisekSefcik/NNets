from numpy.random import randint, randn
import numpy as np
import matplotlib.pyplot as plt


class CGanTrainer:
    def __init__(self, discriminator, generator, combined, batch_size, n_classes, noise_dim):
        self.discriminator = discriminator
        self.generator = generator
        self.combined = combined
        self.BATCH_SIZE = batch_size
        self.HALF_BATCH = int(batch_size / 2)
        self.n_classes = n_classes
        self.noise_dim = noise_dim

    def train_step(self, real_images, real_labels):

        real_loss, _ = self.discriminator.train_on_batch([real_images, real_labels], np.ones((self.HALF_BATCH, 1)))

        input_noise, random_y = self.generate_fake_input(self.HALF_BATCH)
        gen_outs = self.generator.predict([input_noise, random_y])
        fake_loss, _ = self.discriminator.train_on_batch([gen_outs, random_y], np.zeros((self.HALF_BATCH, 1)))

        input_noise_gan, random_y_gan = self.generate_fake_input(self.BATCH_SIZE)
        gan_loss = self.combined.train_on_batch(
            [input_noise_gan, random_y_gan],
            np.ones((self.BATCH_SIZE, 1))
        )
        print(real_loss, fake_loss, gan_loss)

    def generate_fake_input(self, n_samples):
        fake_labels = randint(0, self.n_classes, n_samples)
        x_input = randn(self.noise_dim * n_samples)
        fake_noise = x_input.reshape(n_samples, self.noise_dim)
        return fake_noise, fake_labels

    def save_and_show_plot(self, examples, n, epoch):
        for i in range(self.n_classes * n):
            plt.subplot(self.n_classes, n, 1 + i)
            plt.axis('off')
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def create_fig(self, epoch):
        noise, _ = self.generate_fake_input(self.n_classes * 10)
        labels = np.asarray([x for _ in range(self.n_classes) for x in range(10)])
        images = self.generator.predict([noise, labels])
        images = (images + 1) / 2.0
        self.save_and_show_plot(images, 10, epoch)
