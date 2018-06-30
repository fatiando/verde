# Contributing Guidelines

First off, thank you for considering contributing to our project!
This is a community-driven project, so it's people like you that make it useful
and successful.

We love contributions from community members, just like you!
There are many ways to contribute:

* Submitting bug reports and feature requests
* Writing tutorials or examples
* Improvements to the documentation
* Writing code which can be incorporated into project for everyone to use

If you get stuck at any point you can create an issue on GitHub (look for the
*Issues* tab in the repository) or contact us at one of the other channels
mentioned below.

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point if you are new to version control.
Also, checkout the
[Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
for some guiding principles on how to create high quality scientific software contributions.


## Ground Rules

The goal is to maintain a diverse community that's pleasant for everyone.
**Please be considerate and respectful of others**.
Everyone must abide by our [Code of Conduct](CODE_OF_CONDUCT.md) and we
encourage all to read it carefully.


## What Can I Do?

* Tackle any issue that you wish!
  Some issues are labeled as **"good first issues** to indicate that they are
  beginner friendly, meaning that they don't require extensive knowledge of the
  project.
* Make a tutorial or example of how to do something.
* Provide feedback about how we can improve the project or about your
  particular use case.
* Contribute code you already have. It doesn't need to be perfect! We will help
  you clean things up, test it, etc.


## How Can I Talk to You?

Discussion often happens in the issues and pull requests.
In addition, there is
[Gitter chat room](https://gitter.im/fatiando/fatiando)
for the Fatiando a Terra project as well.


## Reporting a Bug

When creating a new issue, please be as specific as possible.
This helps us reproduce the bug and track down its cause.
Try to include the following:

* Version of the code you were using
* Operating system
* Python installation (Anaconda, system, ETS)
* Full error messages that you got
* Example code that reproduces the problem

Remember: the more information we have, the easier it will be for us to solve
your problem.


## Pull Requests

**Working on your first Pull Request (PR)?**
You can learn how from this *free* video series:

* [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)
* Aaron Meurer's [tutorial on the git workflow](http://www.asmeurer.com/git-workflow/)
* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/).

General guidelines for Pull Requests:

* **First open an issue** describing what you want to do. If there is already
  an issue that matches your PR, leave a comment there instead. This let's us
  better coordinate who is working on what and avoid duplicated efforts.
* Each pull request should consist of a **small** and logical collection of
  changes.
  Larger changes should be broken down into smaller components and integrated
  separately.
  This allows us more chance for discussion and less code to review at each
  time.
  Please submit bug fixes in separate pull requests.
* Describe what your PR changes and why this is a good thing. Be as specific as
  you can. The PR description is how we keep track of the changes made to the
  project over time.
* Do not commit changes to files that are irrelevant to your feature or bugfix
  (eg: .gitignore, IDE project files, etc).
* Write descriptive commit messages.  Chris Beams has written a
  [guide](https://chris.beams.io/posts/git-commit/) on how to write good commit
  messages.
* Be willing to accept criticism and work on improving your code; we don't want
  to break other users' code, so care must be taken not to introduce bugs.
* Be aware that the pull request review process is not immediate, and is
  generally proportional to the size of the pull request.
* **If this is your first contribution**, be sure to add yourself to the
  [list of contributors](AUTHORS.md). We want to make sure we acknowledge the
  hard work you've generously contributed here.


## Setup

We highly recommend using
[Anaconda](https://www.anaconda.com/download/)
and the `conda` package manager.
It will make your life a lot easier!

Once you have forked and clone the repository to your local machine,
create an isolated environment for you to work:

    conda env create

This will install all you need from conda-forge into a separate environment.
See the [`environment.yml`](environment.yml) file for the list of dependencies
and the environment name.
Activate it by running:

    source activate ENVIRONMENT_NAME


## Automation

The `Makefile` provides rules for installing, running the tests and coverage
analysis, running linters, etc.
If you don't want to use `make`, see the [Makefile](Makefile) and copy the
commands you want to run.

Install the source as a development version (it will only be available
inside the environment and changes to the source will take effect without
reinstalling):

    make develop

Run the tests using:

    make test

and the coverage analysis using:

    make coverage

We use [Black](https://github.com/ambv/black) to format the code so we don't have to
think about it. You don't have to worry about formatting the code yourself. Before
committing, run the following to automatically format your code:

    make format

Don't worry if you forget to do it. Our continuous integration systems will warn us and
you can make a new commit with the formatted code.

To check your code for style and common errors (runs `black` and `pylint`):

    make check

Finally, to remove all build files from the repository:

    make clean

There is also a `Makefile` for building the documentation in the `doc` folder:

    cd doc
    make html

You can preview the doc pages in your browser by running:

    make serve

This will serve the docs on a local host. See the command outputs for the URL.


## Code Style and Review

Once you've submitted a Pull Request (PR), at this point you're waiting on us.
You should expect to hear at least a comment within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted
quickly:

* Write a good and detailed description of what the PR does.
* Write tests for the code you wrote/modified.
* Readable code is better than clever code (even with comments).
* Write documentation for your code (docstrings) and leave comments explaining
  the *reason* behind non-obvious things.
* Follow the [PEP8](http://pep8.org) style guide for code and the [numpy
  guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
  for documentation.

Pull requests will automatically have tests run by TravisCI.
This includes running both the unit tests as well as the `flake8` and `pylint`
code linters.
Github will show the status of these checks on the pull request.
Try to get them all passing (green).
If you have any trouble, leave a comment asking for help.


## Testing your code

Automated testing helps ensure that our code is as free of bugs as it can be.
It also lets us know immediately if a change we make breaks any other part of
the code.

All of our test code and data are stored in the `tests` subpackage.
We use the [pytest](https://pytest.org/) framework to run the test suite.

Please write tests for your code so that we can be sure that it won't break any
of the existing functionality.
Tests also help us be confident that we won't break your code in the future.

If you're **new to testing**, see the files in that package for examples of
test code.
**Don't let the tests keep you from submitting code to us!**
If you're not sure how to do this or are having trouble, submit your pull
request anyway.
We will help you create the tests and sort out any kind of problem during code
review.

### Testing plots

We use the [pytest-mpl](https://github.com/matplotlib/pytest-mpl) plugin to
test plot generating code.
Every time the tests are run, `pytest-mpl` compares the generated plots with
known correct ones stored in `verde/tests/baseline`.
To make a plot comparison test, add the `pytest.mark.mpl_image_compare` decorator to
your test function:

```python
@pytest.mark.mpl_image_compare
def test_my_plotting_case():
    "Test that my plotting function works"
    fig = plt.figure()
    plt.plot([0, 1, 2], [3, 4, 5])
    return fig
```

Your test function **must** return the matplotlib figure object and you can only
test one figure per function.

Before you can run your test, you'll need to generate a *baseline* (a correct
version) of your plot.
Run the following from the repository root:

    py.test --mpl-generate-path=baseline verde/tests/NAME_OF_TEST_FILE.py

This will create a `baseline` folder with all the plots generated in your test
file.
Visually inspect the one corresponding to your test function.
If it's correct, copy it (and only it) to `verde/tests/baseline`.
When you run `make test` the next time, your test should be executed and
passing.

Don't forget to commit the baseline image as well.


## Credit

This guide was adapted from the [MetPy Contributing
Guide](https://github.com/Unidata/MetPy/blob/master/CONTRIBUTING.md).
