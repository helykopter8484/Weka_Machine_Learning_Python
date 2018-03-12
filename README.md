## Demonstration of Python Weka Wrapper and Bash Script for Machine_Learning.

This is a small project to demonstrate some code samples, using python, bash and python weka wrapper.
The details of the python weka wrapper (developed originally by Peter Reutemann) can be found at:
https://github.com/fracpete/python-weka-wrapper.
The programs were developed with an application to bioinformatics and machine learning.

### Prerequisites

The scripts are written in Python and use a few other libraries.

* Python
* python-weka-wrapper (>= 0.2.0)
* javabridge (>= 1.0.14)
* matplotlib (optional)
* pygraphviz (optional)
* PIL (optional)
* JDK 1.6+

Libraries can be installed by using the Python Package Index - PIP.

```
pip install <package-name>
```

* #### Bash_Weka_Classification.sh:
A bash script to perform Classification using Weka.
The script takes all the .csv files in the folder and performs classification using weka.
The classifiers used are: J48, J48graft, RandomForest and RandomTree.
10 fold cross validation is performed. The classification output are saved as text files along with the models. The .csv files must be present in the same folder along with this script.

* #### Py_Subprocess_Weka.py:
This program uses the subprocess module to run Weka classifiers (J48, RandomTree and RandomForest).
Other classifiers can be added as required by changing the arguments to the subprocess.
The program runs classifiers on all the .csv files present in the current directory.

* #### Python_Weka_Wrapper_Classification.py:
This program uses the Python Weka Wrapper for Classification.
The Prediction Summary is presented in a text file with the suffix "Prediction.txt"
Overall the code presents the following output: the classification tree, prediction, evaluation and the ROC image. The program takes all the csv files in a folder as an input and performs classification. The values of "Correctly Classified Instances" are presented in a csv file.

* #### Python_Weka_Wrapper_Feature_Selection.py:
This program uses the Python Weka Wrapper for Feature Selection, performed by CfsSubsetEval and BestFirst.
