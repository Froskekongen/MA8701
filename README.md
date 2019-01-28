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
for loading a tensorflow and pytorch module. To load these modules, run
```
source load_tf_modules.sh
```

## Setting up the python environment

To make sure that everything is as expected, we should inspect what the
tensorflow module contains and that it works as expected.

First, we load the modules that we need:
```
source load_tf_modules.sh
```

Then we can run an interactive python environment, `ipython`.
In `ipython`, we check that tensorflow and pytorch is installed
and that we can access a gpu
```
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(tf.__version__)
```
The output should indicate that a GPU is available and which version
of tensorflow we have available.
For torch, we do
```
import torch
print(torch.cuda.is_available())
print(torch.__version__)
```

### Creating a virtual environment
The modules we have loaded contain the main libraries we need
for estimating models using either tensorflow or pytorch. We
may, however, need more libraries, and we want full control
of what we use when we run experiments. For this we use python
`virtualenv`. To create a virtual environment, we run
```
virtualenv model_env
```
To use this environment instead of the "global" environment,
we run
```
source model_env/bin/activate
```
A problem with this environment is that we don't have access
to all the pre-installed packages in the modules we have
loaded with `module load`. To create an environment that
has access to these packages, we create a virtualenv using
the following command instead
```
virtualenv --system-site-packages model_env
```
When in this environment, we can install any package we need
and still have access to what the module has loaded.

To delete a virtualenv (it is common make mistakes the first times),
we delete the folder in which it has been created.
```
rm -rf model_env/
```
It is recommended to create scripts that create the virtualenvs
rather than making them interactively. We do this in order to
create reproducible environments. An example is given in
`create_venv.sh`. Here we install the newest version of keras.

## Scheduling long-running code through SLURM-scripts

For a detailed information on scheduling jobs using `slurm`, see
[this page](https://slurm.schedmd.com/). We will only do a quick
summary here, to get started with scheduling jobs.

To submit a long-running job, we use the `sbatch` command:
```
sbatch some_shell_script.sh
```
will schedule the a job using the commands within `some_shell_script.sh`.

Inside a slurm shell-script, there are some special headers that will
serve as configuration for the scheduled job. These headers are
prefixed by `#SBATCH`.

Typical confugration parameters are which computational resources
to use, as well as output logging, and naming of the job(s).

There are some general guidelines available [here](https://www.hpc.ntnu.no/display/hpc/Getting+Started+on+Idun).

For using GPUs, the following are required
```
#SBATCH --gres=gpu:$N_GPUs
```
where `$N_GPUS` is the number of gpus to require (typically 1).

The `#SBATCH --time=XX:YY:ZZ` configuration sets the time limit for the job in
`XX` hours, `YY` minutes and `ZZ` seconds. When the time limit has hit,
the job will be killed. It's wise to set reasonable time limits for scheduling
purposes.
