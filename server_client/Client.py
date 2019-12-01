# Import socket module
import socket
import time
import os

host = '129.82.44.146'

class Switcher(object):
    def indirect(self, i):
        method_name = 'o_' + str(i)
        method = getattr(self, method_name, lambda: 'Invalid')
        return method()

    def o_1(self):
        return '1'

    def o_2(self):
        print("Option 2 is selected")
        return '2'


def file_sender(name, file_transferring_port):
    global host
    #file_transferring_port = 50581
    time.sleep(4)

    file_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #file_socket.settimeout(10)
    # connect to server on local computer
    try_connection_for_x_times = int(5)
    while(try_connection_for_x_times!=int(0)):
        try:
            file_socket.connect((host, file_transferring_port))
            print("Client connected to the server")
            try_connection_for_x_times = int(0)
        except socket.error as exc:
            print("Failed to connect : %s" % exc)
            try_connection_for_x_times = try_connection_for_x_times - 1
            time.sleep(4)

    file_exist = 1
    while file_exist:
        file_location = input("File Location of " + name + ": ")
        if os.path.isfile(file_location):
            file_exist = 0
        else:
            print("File Doesn't Exist!!!")

    f = open(file_location, 'rb')
    file_data = f.read(1024)
    while (file_data):
        file_socket.send(file_data)
        file_data = f.read(1024)
    f.close()
    file_socket.close()


def Main():
    # local host IP '127.0.0.1'
    global host

    # Define the port on which you want to connect
    port = 50580

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # connect to server on local computer
    s.connect((host, port))

    # message you send to server
    list = []
    message = ""
    while True:
        option = input('1. Add a Expert\n 2.Evaluation on cluster\n 3. Exit\n')
        optionSelected = Switcher()
        message = optionSelected.indirect(option)

        if message == '1':
            s.send(message.encode('ascii'))
            port = s.recv(1024)
            string_port = str(port.decode('ascii'))
            list_port = string_port.split(":")
            port = list_port[-2]
            print("Sending data to server at port " + str(port))
            file_sender("MOE", int(port))
            file_sender("TOE", int(port))
            file_sender("SOE", int(port))
            print("File Forwarded to the client")

        if message == '2':
            s.send(message.encode('ascii'))
            port = s.recv(1024)
            file_sender("MOE", int(port))

        if message == '3':
            break;

        # ask the client whether he wants to continue
        ans = input('\nDo you want to continue(y/n) : ')
        if ans == "y":
            continue
        else:
            break
    # close the connection
    s.close()


if __name__ == '__main__':
    Main()
