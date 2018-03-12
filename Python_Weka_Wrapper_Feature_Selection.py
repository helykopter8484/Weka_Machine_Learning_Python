
import os
import sys
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


def Feature_Selection(infile):
    directory = os.getcwd() + '/'
    csvpath = directory + infile

    jvm.start(packages=True, max_heap_size="4g")
    print "\n\n"
    print "Loaded file: ", infile
    csvloader = Loader(classname="weka.core.converters.CSVLoader")
    csvdata = csvloader.load_file(csvpath)

    remover = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", " 1"])
    remover.inputformat(csvdata)
    filtered_data = remover.filter(csvdata)
    filtered_data.class_is_last()

    search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
    evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    attribs = AttributeSelection()
    attribs.search(search)
    attribs.evaluator(evaluator)
    attribs.select_attributes(filtered_data)
    print "Summary of Attribute Selection: "
    print attribs.results_string
    jvm.stop()
    return

if len(sys.argv) != 2:
    print """Performing Classification using Python Weka Wrapper:
    http://pythonhosted.org/python-weka-wrapper/index.html#
    Usage: Provide the csv filename as the argument.
    The csv file should be present in the working directory."""
    sys.exit()
else:
    Feature_Selection(sys.argv[1])
