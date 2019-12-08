from numpy.random import randint, randn
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib

matplotlib.rcParams['figure.figsize'] = [20, 20]

class CGanTrainer:
    def __init__(self, discriminator, generator, combined, batch_size, n_classes, noise_dim):
        self.discriminator = discriminator
        self.generator = generator
        self.combined = combined
        self.BATCH_SIZE = batch_size
        self.HALF_BATCH = int(batch_size / 2)
        self.n_classes = n_classes
        self.noise_dim = noise_dim

    def train_step(self, real_images, real_labels, batch_no):

        real_loss, _ = self.discriminator.train_on_batch([real_images, real_labels], np.ones((self.HALF_BATCH, 1)))

        input_noise, random_y = self.generate_fake_input(self.HALF_BATCH)
        gen_outs = self.generator.predict([input_noise, random_y])
        fake_loss, _ = self.discriminator.train_on_batch([gen_outs, random_y], np.zeros((self.HALF_BATCH, 1)))

        input_noise_gan, random_y_gan = self.generate_fake_input(self.BATCH_SIZE)
        gan_loss = self.combined.train_on_batch(
            [input_noise_gan, random_y_gan],
            np.ones((self.BATCH_SIZE, 1))
        )
        print(batch_no, real_loss, fake_loss, gan_loss)
        return real_loss, fake_loss, gan_loss

    def generate_fake_input(self, n_samples):
        fake_labels = randint(0, self.n_classes, n_samples)
        x_input = randn(self.noise_dim * n_samples)
        fake_noise = x_input.reshape(n_samples, self.noise_dim)
        return fake_noise, fake_labels

    def save_and_show_plot(self, examples, n, m, epoch):
        fig, ax = plt.subplots()
        for i in range(n * m):
            plt.subplot(m, n, 1 + i)
            plt.axis('off')
            if examples[i].shape[2] > 1:
                plt.imshow(examples[i])
            else:
                plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.show()
        return buf

    def create_fig(self, epoch):
        noise, labels = self.generate_fake_input(25)
        # labels = np.asarray([x for _ in range(self.n_classes) for x in range(10)])
        images = self.generator.predict([noise, labels])
        images = (images + 1) / 2.0
        return self.save_and_show_plot(images, 5, 5, epoch)


def generate_image_show(generator, noise_dim, label):
    x_input = randn(noise_dim)
    fake_noise = x_input.reshape(1, noise_dim)
    image = generator.predict([fake_noise, np.array([label])])
    image = (image + 1) / 2.0
    if image.shape[3] > 1:
        plt.imshow(image[0])
    else:
        plt.imshow(image[0, :, :, 0], cmap='gray_r')
    return image
