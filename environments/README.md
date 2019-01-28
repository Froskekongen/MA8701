# What are environments?

We want our computations to run with deterministic software.
What this means, is that when we run some experiments,
we want to know which versions of all libraries, software,
and possibly also hardware we run on. This is to ensure
that whenever we run some code, we get what we expect.

On a cluster, such as the epic cluster, many users need to run
their experiment using libraries, etc..., that they know
will yield the expected results. It is therefore convenient
to have a system that lets the user change between such
configuration easily. These specific configurations that are
needed to make specific versions of libraries is the environment.

How the environment is configured is often done through
[environment variables](https://encyclopedia2.thefreedictionary.com/Environment+(computer+science)).
