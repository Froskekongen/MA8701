from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras.models import Model, Sequential
from keras import optimizers
from datetime import datetime
import json
from pathlib import Path


def create_simple_convnet():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    return model


def load_resnet50():
    resnet50_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
    return resnet50_model


if __name__ == '__main__':
    """
    Example for transfer learning is taken from
    https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument("--test_run", default=True)
    args = parser.parse_args()
    start = datetime.now()

    with open(args.config_path) as ff:
        config = json.load(ff)
    opt_config = config.pop("optimizer")

    glob_file_pattern = "**/*.{0}".format(config["input_filetype"])
    n_train_examples = len(list(Path(config["train_path"]).glob(glob_file_pattern)))
    n_classes = len(list(Path(config["train_path"]).glob("*/**")))
    n_valid_examples = len(list(Path(config["valid_path"]).glob(glob_file_pattern)))
    if args.test_run is True:
        n_steps = 10
        n_steps_valid = 5
    else:
        n_steps = n_train_examples // config["batch_size"]
        n_steps_valid = n_valid_examples // config["batch_size"]
    img_size = (224, 224)

    print('Doing {0} steps of training and {1} steps of validation.'.format(n_steps, n_steps_valid))
    # See https://keras.io/preprocessing/image/
    # for details.
    datagen_train = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    datagen_test = ImageDataGenerator(rescale=1. / 255)

    train_images = datagen_train.flow_from_directory(config["train_path"],
                                                     target_size=img_size,
                                                     class_mode='categorical',
                                                     classes=['dogs', 'cats'],
                                                     batch_size=config["batch_size"])

    valid_images = datagen_test.flow_from_directory(config["valid_path"],
                                                    target_size=img_size,
                                                    class_mode='categorical',
                                                    classes=['dogs', 'cats'],
                                                    batch_size=config["batch_size"])

    if config["model_type"] == "simple_convnet":
        print("Using simple convnet.")
        loaded_model = create_simple_convnet()
    elif config["model_type"] == "resnet50":
        print("Using pretrained resnet.")
        loaded_model = load_resnet50()
        for layer in loaded_model.layers:
            layer.trainable = False

    x = loaded_model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=loaded_model.input, outputs=predictions)
    opt = getattr(optimizers, opt_config["type"])(**opt_config["params"])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Fitting a model.')
    model.fit_generator(train_images,
                        steps_per_epoch=n_steps,
                        epochs=3,
                        verbose=1,
                        validation_data=valid_images,
                        validation_steps=n_steps_valid,
                        max_queue_size=20,
                        shuffle=True,
                        initial_epoch=0)
    model.save(config["model_id"])
    finish = datetime.now()
    td = finish - start
    elapsed_mins = td.total_seconds() / 60.
    print('Total time {0}'.format(elapsed_mins))
