from setuptools import setup

setup(
    name='titanic-kaggle',
    version='0.0.1',
    description='Titanic Kaggle task https://www.kaggle.com/c/titanic',
    url='https://github.com/vogdb/titanic-kaggle/',
    license='MIT',
    python_requires='>3.5',
    packages=['titanic_kaggle'],
    package_dir={'titanic_kaggle': 'titanic_kaggle'},
    package_data={'titanic_kaggle': ['dataset']},
)
