#!/bin/bash
# Script to setup python virtual environment, and install ipython kernel, to be
# used for 01-adversarial.

# Check whether virtualenv is installed
if ! type virtualenv > /dev/null 2>&1; then
    echo "Please install virtualenv. For instance, do:"
    echo "  $ pip install virtualenv [--user]"
    echo "See also: https://virtualenv.pypa.io/en/latest/installation/"
    return 1
fi

# Determine python executable to use
declare -a pythonexes=("python2.7" "python2" "python")
for pythonexe in "${pythonexes[@]}"; do
    if eval "$pythonexe -V > /dev/null 2>&1"; then
	break
    fi
done

# Check that we're using python2
pythonversion="$(eval $pythonexe -V 2>&1 | sed 's/Python *//g;s/ .*//g')"
if [[ ! $pythonversion =~ ^2\..* ]]; then
    echo "Python version not supported:"
    eval $pythonexe -V
    return 1
fi

# Setup virtual environment
name="env-01-adversarial"
echo "Creating virtual environment $name with python executable $pythonexe."
echo "---------------------"
virtualenv -p $pythonexe $name
source $name/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
ipython kernel install --user --name=$name
deactivate
echo "---------------------"
echo "To use this virtual environment in the Jupyter Notebook, open the .ipynb file and select \"Kernel > Change Kernel > $name\" in the toolbar."
