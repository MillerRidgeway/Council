import globals
import os
from subprocess import PIPE, Popen
import subprocess
import re


# thread function
def hdfs_pusher(storedFileName):
    hdfs_path = os.path.join(os.sep, 'user', globals.user_name, globals.default_hdfs_path)
    # put file into hdfs
    put = Popen(["/usr/local/hadoop/bin/hadoop", "fs", "-put", storedFileName, hdfs_path], stdin=PIPE, bufsize=-1)
    put.communicate()
    f = open("file_uploaded.txt", "a")
    f.write(hdfs_path + storedFileName[storedFileName.rfind("/"):] + "\n")
    f.close()


# thread function
def file_checker():
    args = "/usr/local/hadoop/bin/hadoop fs -ls " + globals.default_hdfs_path + " | awk '{print $8}'"
    file_checker = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    s_output, s_err = file_checker.communicate()
    all_dart_dirs = s_output.split()
    max_value = int(-1)
    for values in all_dart_dirs:
        temp = re.findall(r'\d+', str(values))[-1]
        if int(temp) > max_value:
            max_value = int(temp)
    return max_value
