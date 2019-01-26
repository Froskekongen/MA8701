# MA8701

This repository contains various resources that may help in MA8701.
In particular, we start by setting up a python environment that
can be used for estimating deep learning models on the epic cluster.

## Login

We use `ssh` for login, and the correct login server is `epic.hpc.ntnu.no`

Running

```
ssh $NTNU_USERNAME@epic.hpc.ntnu.no
```

should let you log into the cluster, provided that you are enrolled
in MA8701.

## Modules

The compute clusters use [EasyBuild](https://github.com/easybuilders/easybuild)
for managing multiple, possibly incompatible environments.

This makes it easy for us to load pre-configured environments that have
tools and libraries ready to use.

We load a module by running
```
module load $MODULE_NAME
```

The `$MODULE_NAME` will in most cases be name that semantically makes sense.

To see the available modules, simply run
```
module spider
```

And to search for a specific module, run
```
module spider $MODULE_NAME
```

Before loading a module, it is a good idea to make sure that the environment
is clean. Therefore, run
```
module purge
```
before loading any modules.


In this repository, the file `load_tf_modules.sh` contains what is needed
for loading a tensorflow and pytorch module. 

## Setting up the python environment


## Running code interactively


## Scheduling long-running code through SLURM-scripts
