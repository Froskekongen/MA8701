## Data generators that don'l load all data in memory

When dealing with very large datasets, it's important
to be able to iterate through the dataset withoug loading
all the data in memory. Simply, because the data may not
fit into memory.

In Keras there is some functionality for this for images.
See [this parge](https://keras.io/preprocessing/image/) for
API details. In this example we will use `ImageDataGenerator`
from Keras.

```
from keras.preprocessing.image import ImageDataGenerator

```

For the project, if you want to use all the image data,
combining several datagenerators may be needed.
