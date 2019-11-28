# import socket programming library
import socket
import wordcount
import integrator
# import thread module
from _thread import *
import threading
from subprocess import PIPE, Popen
import os
import time
import subprocess
import select
import re


from pyspark.sql import SparkSession


spark_session = SparkSession.builder.appName('MOE').getOrCreate()

#print_lock = threading.Lock()

user_name = "ms2718"

default_hdfs_path = "/Ass3/"


current_file_number = int(0)

def file_name_generator(name):
    global current_file_number
    current_file_number += 1
    file_name = "/tmp/" + name
    file_name = file_name + str(current_file_number)
    return file_name


def file_reciever(name):
    host = ""
    port = 50581
    fileReciever = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fileReciever.bind((host, port))
    fileReciever.listen(5)
    print("Socket is waiting for Client to send a file ....")

    filesocket, address = fileReciever.accept()
    #storedFileName = file_name_generator(name)
    storedFileName = name
    f = open(storedFileName, 'wb')

    # receive data and write it to file
    print("Writing a file")
    file_data = filesocket.recv(1024)
    while (file_data):
        f.write(file_data)
        file_data = filesocket.recv(1024)
    f.close()
    filesocket.close()
    fileReciever.close()
    return storedFileName


# thread function
def hdfs_pusher(storedFileName):
    global user_name
    global default_hdfs_path
    hdfs_path = os.path.join(os.sep, 'user', user_name, default_hdfs_path)

    # put file into hdfs
    put = Popen(["/usr/local/hadoop/bin/hadoop", "fs", "-put", storedFileName, hdfs_path], stdin=PIPE, bufsize=-1)
    put.communicate()
    #return storedFileName[storedFileName.rfind("//"):]
    f = open("file_uploaded.txt", "a")
    f.write(hdfs_path + storedFileName[storedFileName.rfind("/"):] + "\n")
    f.close()
    #return hdfs_path + storedFileName[storedFileName.rfind("/"):]


# thread fuction
def threaded(c):
    global spark_session
    while True:

        # data received from client i.e 1. Add a Expert OR 2.Evaluation on cluster
        data = c.recv(1024)
        #print("Option recieved "  + data)
        if not data:
            print('Client Disconnected')
            break

        # New Expert need to be added
        if data == b"1":
            print("Request for inserting new node")
            list_of_files_to_be_pushed_to_hdfs = []
            # file_name_generator to generate file names
            generated_name = file_name_generator("MOE")
            # All File names will have same suffix
            list_of_files_to_be_pushed_to_hdfs.append(generated_name)
            list_of_files_to_be_pushed_to_hdfs.append(generated_name[:generated_name.rfind("/")+1] + "TOE" + re.findall(r'\d+', generated_name)[-1])
            list_of_files_to_be_pushed_to_hdfs.append(
                generated_name[:generated_name.rfind("/")+1] + "SOE" + re.findall(r'\d+', generated_name)[-1])

            file_reciever(list_of_files_to_be_pushed_to_hdfs[0])
            file_reciever(list_of_files_to_be_pushed_to_hdfs[1])
            file_reciever(list_of_files_to_be_pushed_to_hdfs[2])

            for file in list_of_files_to_be_pushed_to_hdfs:
                # After the files are received start pushing it to hdfs
                start_new_thread(hdfs_pusher, (file,))

        # To evaluate a input
        if data == b"2":
            print("Request for Evaluation")
            file_to_be_evaluated = file_name_generator("Evaluation")
            #list_of_files_to_be_pushed_to_hdfs.append()
            file_reciever(file_to_be_evaluated)
            hdfs_pusher(file_to_be_evaluated)
            # Evaluation code here
            global default_hdfs_path
            wordcount.word_count(spark_session, default_hdfs_path + file_to_be_evaluated[file_to_be_evaluated.rfind("/"):])
            #for file in list_of_files_to_be_pushed_to_hdfs:
                # push the recieved file in the hdfs
                #hdfs_file_path = hdfs_pusher(file_to_be_evaluated)
                # specify the file prefix and call the function
                #if "Evaluation" in  hdfs_file_path:
                    # Call the function - replace with the evaluation code
                 #   wordcount.word_count(spark_session, hdfs_file_path)

    c.close()


def Main():
    host = ""

    # Port for listening Client Request
    port = 50580

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))

    print("socket binded to port", port)

    # put the socket into listening mode
    s.listen(5)

    print("socket is waiting for Client to Join ....")

    # Checks if all 3 files are ready
    start_new_thread(integrator.new_expert_trainer,(spark_session,))

    # a forever loop for n client connections
    while True:
        c, addr = s.accept()

        print('Server is now connected to :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))
    s.close()


if __name__ == '__main__':
    Main()

