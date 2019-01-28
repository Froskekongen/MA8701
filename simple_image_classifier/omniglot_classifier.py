from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
from datetime import datetime

if __name__ == '__main__':
    """
    Example for transfer learning is taken from
    https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--image_folder',
                        default='/lustre1/projects/fs_ma8701_1/omniglot_processed')
    parser.add_argument('--weights', default='imagenet')
    parser.add_argument('--test_run', default=True)
    args = parser.parse_args()
    start = datetime.now()

    if args.test_run is True:
        n_steps = 5
        n_steps_valid = 2
    else:
        n_steps = 52
        n_steps_valid = 13

    print('Doing {0} steps of training and {1} steps of validation.'.format(n_steps, n_steps_valid))
    # See https://keras.io/preprocessing/image/
    # for details.
    datagen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen_test = ImageDataGenerator(rescale=1. / 255)

    train_images = datagen_train.flow_from_directory(args.image_folder + '/train',
                                                     target_size=(224, 224))

    valid_images = datagen_test.flow_from_directory(args.image_folder + '/valid',
                                                    target_size=(224, 224))

    resnet50_model = ResNet50(weights=args.weights, include_top=False, pooling='avg')

    x = resnet50_model.output
    x = Dense(256, activation='relu')(x)
    predictions = Dense(136, activation='softmax')(x)
    for layer in resnet50_model.layers:
        layer.trainable = False

    model = Model(inputs=resnet50_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Fitting a model.')
    model.fit_generator(train_images,
                        steps_per_epoch=n_steps,
                        epochs=12,
                        verbose=1,
                        callbacks=None,
                        validation_data=valid_images,
                        validation_steps=n_steps_valid,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0)
    model.save('resnet_omniglot.h5')
    finish = datetime.now()
    td = finish - start
    elapsed_mins = td.total_seconds() / 60.
    print('Total time {0}'.format(elapsed_mins))
