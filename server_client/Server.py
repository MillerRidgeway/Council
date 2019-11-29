# import socket programming library
import socket
import wordcount
import integrator
import globals
import hdfs_uploader

from _thread import *
import threading

import os
import time
import subprocess
import select
import re

current_file_number = int(0)

def file_name_generator(name):
    global current_file_number
    current_file_number += 1
    file_name = "/tmp/" + name
    file_name = file_name + str(current_file_number)
    return file_name


def file_reciever(name):
    fileReciever = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fileReciever.bind((globals.host, globals.file_receiving_port))
    fileReciever.listen(5)
    print("Socket is waiting for Client to send a file ....")

    filesocket, address = fileReciever.accept()
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


def threaded(c):
    while True:

        # data received from client i.e 1. Add a Expert OR 2.Evaluation on cluster
        data = c.recv(1024)
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
                start_new_thread(hdfs_uploader.hdfs_pusher, (file,))

        # To evaluate a input
        if data == b"2":
            print("Request for Evaluation")
            file_to_be_evaluated = file_name_generator("Evaluation")
            #list_of_files_to_be_pushed_to_hdfs.append()
            file_reciever(file_to_be_evaluated)
            hdfs_uploader.hdfs_pusher(file_to_be_evaluated)
            wordcount.word_count(globals.spark_session, globals.default_hdfs_path + file_to_be_evaluated[file_to_be_evaluated.rfind("/"):])


    c.close()


def Main():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((globals.host, globals.client_request_port))

    print("socket binded to port", globals.client_request_port)

    # put the socket into listening mode
    s.listen(5)

    print("socket is waiting for Client to Join ....")

    # Checks if all 3 files are ready
    start_new_thread(integrator.new_expert_trainer,(globals.spark_session,))

    # a forever loop for n client connections
    while True:
        c, addr = s.accept()

        print('Server is now connected to :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))
    s.close()


if __name__ == '__main__':
    Main()

