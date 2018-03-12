
import os
import sys
import subprocess


def GetFilenames(directory):
    for root, dirs, files in os.walk(directory):
        CSV_files = []
        for filename in files:
            if filename.endswith(('.csv')):
                CSV_files.append(filename)
        return CSV_files


def Run_J48(directory, filelist):
    for name in filelist:
        fid = str(name)
        fpth = str(name[0:-4]) + '_J48'
        pth = directory + fid
        WEKA = ["java", "-Xmx4G", "weka.classifiers.trees.J48",
                "-t", pth, "-d", fpth + ".model"]
        process = subprocess.Popen(WEKA, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        f = open(directory + fpth + '.txt', 'w')
        f.write(stdout)
        f.close()
    return stdout, stderr


def Run_RandomTree(directory, filelist):
    for name in filelist:
        fid = str(name)
        fpth = str(name[0:-4]) + '_J48graft'
        pth = directory + fid
        WEKA = ["java", "-Xmx4G", "weka.classifiers.trees.RandomTree",
                "-t", pth, "-d", fpth + ".model"]
        process = subprocess.Popen(WEKA, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        f = open(directory + fpth + '.txt', 'w')
        f.write(stdout)
        f.close()
    return stdout, stderr


def Run_RandomForest(directory, filelist):
    for name in filelist:
        fid = str(name)
        fpth = str(name[0:-4]) + '_RF'
        pth = directory + fid
        WEKA = ["java", "-Xmx4G", "weka.classifiers.trees.RandomForest",
                "-t", pth, "-d", fpth + ".model"]
        process = subprocess.Popen(WEKA, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        f = open(directory + fpth + '.txt', 'w')
        f.write(stdout)
        f.close()
    return stdout, stderr


if __name__ == '__main__':
    print """
    This program uses the subprocess module to run Weka classifiers, namely
    J48, RandomTree and RandomForest. The classification results and the
    resulting models are saved to the disk. The program runs classifiers on
    all the .csv files present in the current directory.
    Important: Please export the weka classpath before usage.
    TRYING TO PERFORM CLASSIFICATION NOW..."""
    directory = os.getcwd() + '/'
    CSVList = GetFilenames(directory)
    Run_J48(directory, CSVList)
    Run_RandomTree(directory, CSVList)
    Run_RandomForest(directory, CSVList)
    print "=" * 100
    print "Performed Classification on: -"
    for item in CSVList:
        print item
    sys.exit()
