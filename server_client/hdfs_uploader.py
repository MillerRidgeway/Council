import globals
import os
from subprocess import PIPE, Popen


# thread function
def hdfs_pusher(storedFileName):
    hdfs_path = os.path.join(os.sep, 'user', globals.user_name, globals.default_hdfs_path)
    # put file into hdfs
    put = Popen(["/usr/local/hadoop/bin/hadoop", "fs", "-put", storedFileName, hdfs_path], stdin=PIPE, bufsize=-1)
    put.communicate()
    f = open("file_uploaded.txt", "a")
    f.write(hdfs_path + storedFileName[storedFileName.rfind("/"):] + "\n")
    f.close()
