# Contributing to trajectorytools

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, trajectorytools version and python version. Whenever 
possible, please also include a brief, self-contained code example 
demonstrating the problem.

## Contributing code

Thanks for your interest in contributing code to trajectorytools!

We gladly welcome pull requests, normally to the develop branch. 

We try very hard to follow good coding practices. This includes:

+ We do not add commits directly to master and develop, only pull requests
+ We follow [PEP 8 recommendations](https://www.python.org/dev/peps/pep-0008/),
including the line length limit of 79 for python code and 72 for docs/comments.
+ We use an autoformatter called [black](https://github.com/psf/black) to have
a uniform code look. Before submitting pull requests, we would appreciate if 
you autoformat the code using the following command: `black -l 79 .`
+ Whenever possible, we create tests for any new code.
+ We are keeping a [CHANGELOG](../CHANGELOG.md)

If you are like us some years ago and have problems following these rules,
we are happy to help you implement them during the pull request process.

## Attribution

This CONTRIBUTING file is highly influenced by the contributing guidelines in
[numpy](https://github.com/numpy/numpy/edit/master/.github/CONTRIBUTING.md)
