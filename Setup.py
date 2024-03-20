import os
import platform
import subprocess
from setuptools import setup

def start_virtualenv():
    # Check the operating system
    if platform.system() == 'Windows':
        # Windows command to activate virtual environment
        activate_cmd = 'venv\\Scripts\\activate.bat'
    else:
        # Linux command to activate virtual environment
        activate_cmd = 'source venv/bin/activate'

    # Start the virtual environment
    subprocess.call(activate_cmd, shell=True)

# Start the virtual environment
start_virtualenv()

# Read the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Your setup configuration goes here
setup(
    name='your_package',
    version='1.0',
    packages=['your_package'],
    install_requires=requirements,
    # other setup options...
)
