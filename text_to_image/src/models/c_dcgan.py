from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class GAN:
    def __init__(self, image_shape, noise_dim=100, n_classes=10):
        self.image_shape = image_shape
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.optimizer = Adam(0.0001, beta_1=0.5)
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.combined = self.get_combined()

    def get_generator(self):
        # Input for label
        input_label = layers.Input(shape=(1,))
        label_flow = layers.Embedding(self.n_classes, 50)(input_label)
        n_nodes = 7 * 7
        label_flow = layers.Dense(n_nodes)(label_flow)
        label_flow = layers.Reshape((7, 7, 1))(label_flow)
        # Input for noise
        input_noise = layers.Input(shape=(self.noise_dim,))
        n_nodes = 256 * 7 * 7
        noise_flow = layers.Dense(n_nodes)(input_noise)
        noise_flow = layers.LeakyReLU(alpha=0.2)(noise_flow)
        noise_flow = layers.Reshape((7, 7, 256))(noise_flow)
        # Merge noise and label
        merge = layers.Concatenate()([noise_flow, label_flow])
        # Define generator
        gen = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(merge)
        gen = layers.LeakyReLU(alpha=0.2)(gen)
        gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
        gen = layers.LeakyReLU(alpha=0.2)(gen)
        output_layer = layers.Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([input_noise, input_label], output_layer)
        return model

    def get_discriminator(self):
        # Input for label
        input_label = layers.Input(shape=(1,))
        label_flow = layers.Embedding(self.n_classes, 50)(input_label)
        n_nodes = self.image_shape[0] * self.image_shape[1]
        label_flow = layers.Dense(n_nodes)(label_flow)
        label_flow = layers.Reshape((self.image_shape[0], self.image_shape[1], 1))(label_flow)
        # Input for image
        input_image = layers.Input(shape=self.image_shape)
        # Merge image and label
        merge = layers.Concatenate()([input_image, label_flow])
        # Define discriminator
        disc = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(merge)
        disc = layers.LeakyReLU(alpha=0.2)(disc)
        disc = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(disc)
        disc = layers.LeakyReLU(alpha=0.2)(disc)
        disc = layers.Flatten()(disc)
        disc = layers.Dropout(0.4)(disc)
        output_layer = layers.Dense(1, activation='sigmoid')(disc)
        model = Model([input_image, input_label], output_layer)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        return model

    def get_combined(self):
        self.discriminator.trainable = False
        gen_noise, gen_label = self.generator.input
        gen_output = self.generator.output
        gan_output = self.discriminator([gen_output, gen_label])
        model = Model([gen_noise, gen_label], gan_output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, beta_1=0.5)
        )
        return model


class GANColor:
    def __init__(self, image_shape, noise_dim=100, n_classes=10):
        self.image_shape = image_shape
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.optimizer = Adam(0.0001, beta_1=0.5)
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.combined = self.get_combined()

    def get_generator(self):
        # Input for label
        input_label = layers.Input(shape=(1,))
        label_flow = layers.Embedding(self.n_classes, 50)(input_label)
        n_nodes = 8 * 8
        label_flow = layers.Dense(n_nodes)(label_flow)
        label_flow = layers.Reshape((8, 8, 1))(label_flow)
        # Input for noise
        input_noise = layers.Input(shape=(self.noise_dim,))
        n_nodes = 256 * 8 * 8
        noise_flow = layers.Dense(n_nodes)(input_noise)
        noise_flow = layers.LeakyReLU(alpha=0.2)(noise_flow)
        noise_flow = layers.Reshape((8, 8, 256))(noise_flow)
        # Merge noise and label
        merge = layers.Concatenate()([noise_flow, label_flow])
        # Define generator
        gen = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(merge)
        gen = layers.LeakyReLU(alpha=0.2)(gen)
        gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
        gen = layers.LeakyReLU(alpha=0.2)(gen)
        gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
        gen = layers.LeakyReLU(alpha=0.2)(gen)
        output_layer = layers.Conv2D(3, (7, 7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([input_noise, input_label], output_layer)
        return model

    def get_discriminator(self):
        # Input for label
        input_label = layers.Input(shape=(1,))
        label_flow = layers.Embedding(self.n_classes, 50)(input_label)
        n_nodes = self.image_shape[0] * self.image_shape[1]
        label_flow = layers.Dense(n_nodes)(label_flow)
        label_flow = layers.Reshape((self.image_shape[0], self.image_shape[1], 1))(label_flow)
        # Input for image
        input_image = layers.Input(shape=self.image_shape)
        # Merge image and label
        merge = layers.Concatenate()([input_image, label_flow])
        # Define discriminator
        disc = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(merge)
        disc = layers.LeakyReLU(alpha=0.2)(disc)
        disc = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(disc)
        disc = layers.LeakyReLU(alpha=0.2)(disc)
        disc = layers.Flatten()(disc)
        disc = layers.Dropout(0.4)(disc)
        output_layer = layers.Dense(1, activation='sigmoid')(disc)
        model = Model([input_image, input_label], output_layer)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        return model

    def get_combined(self):
        self.discriminator.trainable = False
        gen_noise, gen_label = self.generator.input
        gen_output = self.generator.output
        gan_output = self.discriminator([gen_output, gen_label])
        model = Model([gen_noise, gen_label], gan_output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, beta_1=0.5)
        )
        return model