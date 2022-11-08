import os

from dotenv import load_dotenv
from setuptools import setup, find_packages

load_dotenv('env.txt')

VERSION = os.getenv('VERSION')

with open('README.md', encoding="utf8") as f:
    readme = f.read()

setup(
    name='icd_classifier',
    version=VERSION,
    description='ICD disease code classifier',
    long_description=readme,
    packages=find_packages()
)
