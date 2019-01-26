from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--image_folder', default='/home/erlenda/omniglot/python/images_background_small1')
    args = parser.parse_args()

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    image_iter = datagen.flow_from_directory(args.image_folder)

    for iii, (x, y) in enumerate(image_iter):
        if iii > 5:
            break
        print(y)
