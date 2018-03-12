#!/bin/bash

echo "
This bash script takes all the .csv files in the folder and performs classification using weka.
The classifiers used are: J48, J48graft, RandomForest and RandomTree. 10 fold cross validation 
is performed. The classification output are saved as text files along with the models.
The .csv files must be present in the same folder along with this script.
"
# Defult values
CV=10		# 10 fold cross validation, change here if required.
files=*.csv	# List of files to be processed.

for inp in $files; do

	outJ48="${tmp2}_J48_Result.txt"
	modelJ48="${tmp2}.J48_model"
	
	outJ48graft="${tmp2}_J48_graft_Result.txt"
	modelJ48graft="${tmp2}.J48_graft_model"
	
	outRF="${tmp2}_RF_Result.txt"
	modelRF="${tmp2}.RF_model"
	
	outRT="${tmp2}_RT_Result.txt"
	modelRT="${tmp2}.RT_model"
	
	# J48 Classifier:
	java -Xmx2G -cp /usr/share/java/java_cup.jar:/usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier           -t $inp \
	-x $CV \
	-d $modelJ48 \
	-F "weka.filters.unsupervised.attribute.Remove -R 13" \
	-W weka.classifiers.trees.J48 -- -C 0.25 -M 2 \
	>> $outJ48
	
	# J48graft Classifier
	java -Xmx2G -cp /usr/share/java/java_cup.jar:/usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier           -t $inp \
	-x $CV \
	-d $modelJ48graft \
	-F "weka.filters.unsupervised.attribute.Remove -R 13" \
	-W weka.classifiers.trees.J48graft -- -C 0.25 -M 2 \
	>> $outJ48graft
	
	# Random Forest Classifier:
	java -Xmx2G -cp /usr/share/java/java_cup.jar:/usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier           -t $inp \
	-x $CV \
	-d $modelRF \
	-F "weka.filters.unsupervised.attribute.Remove -R 13" \
	-W weka.classifiers.trees.RandomForest -- -I 100 -K 0 -S 1 \
	>> $outRF
	
	# Random Tree Classifier:
	java -Xmx2G -cp /usr/share/java/java_cup.jar:/usr/share/java/weka.jar weka.classifiers.meta.FilteredClassifier           -t $inp \
	-x $CV \
	-d $modelRT \
	-F "weka.filters.unsupervised.attribute.Remove -R 13" \
	-W weka.classifiers.trees.RandomTree -- -K 0 -M 1.0 -S 1 \
	>> $outRT

	printf "%0.s-" {1..50}
	printf "\n"
	printf "%0.s-" {1..50}
	printf "\n"
    echo "DONE: $inp"
done
