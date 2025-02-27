import argparse
import logging
import os
import shutil
import subprocess

logging.basicConfig(level=logging.INFO)

subprocess.run(["pip", "install", "--upgrade", "pip"])

os.chdir("/opt/ml/processing/apt/")

command_one = fr"""
pip install -r requirements.txt
pip install -e .
python train_cyclegan.py
"""
subprocess.run(command_one, shell=True, check=True)