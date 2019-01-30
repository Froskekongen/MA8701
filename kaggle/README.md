# Using kaggle to download datasets

The most convenient way for downloading datasets from kaggle is to use
the kaggle command line tools. To install the kaggle command line tools
run
```
pip3 install kaggle
```

## Creating a kaggle API key

Before you can use the kaggle command line tools, it is necessary to make
the kaggle API key available in the home folder.

Go to `https://www.kaggle.com/USERNAME/account`. Here you can obtain an API-key.
Copy the contents of the API key to `YOUR_HOME_FOLDER/.kaggle/kaggle.json`.
Now commands like
```
kaggle competitions download -c dogs-vs-cats
```
will work.

Note that you will need to accept the rules for the competition on the kaggle
website before you can use the tool to download the data.
