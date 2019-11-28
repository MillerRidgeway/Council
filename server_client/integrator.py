import time
import subprocess
import select
import re
import os
import wordcount

default_hdfs_path = "/Ass3/"

class file_dictionary(dict):
    # __init__ function
    def __init__(self):
        self = dict()
    def add(self, key, value):
        self[key] = value
    def get(self, key):
        if key != "":
            if key in self:
                return self[key]
            else:
                return -1
    def remove(self,key):
        if key in self:
            self.pop(key)


def tail_file(thefile):
    thefile.seek(0, 2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

def new_expert_trainer(spark_session):
    global default_hdfs_path
    print("inside funciton")
    filename = "file_uploaded.txt"
    if os.path.isfile(filename):
        os.remove(filename)
    f = open(filename, "w+")
    f.close()
    logfile = open(filename, "r")
    loglines = tail_file(logfile)
    dict_obj = file_dictionary()
    for line in loglines:
                values = str(re.findall(r'\d+', str(line.strip()))[-1])
                print("Reading Value " + values + " in dictionary" + str(dict_obj.get(values)))
                if dict_obj.get(str(values)) == -1:
                    print("New Value")
                    dict_obj.add(re.findall(r'\d+', values)[-1], 1)
                    print(dict_obj)
                else :
                    print("Already Exist Value")
                    temp = dict_obj.get(values)
                    print("current value of "+ values + " is "+ str(temp))
                    if temp+1 < 3:
                        dict_obj.add(values, int(temp)+1)
                        print(dict_obj)
                    else:
                        # run the MOE updating
                        dict_obj.remove(values)
                        print("3 files are present, running the update")

                        wordcount.word_count(spark_session, default_hdfs_path + "MOE" + values)
