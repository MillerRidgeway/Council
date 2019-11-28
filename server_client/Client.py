# Import socket module
import socket
import time

host = '127.0.0.1'

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

def file_sender(name):
    global host
    file_transferring_port = 50581
    time.sleep(4)

    file_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to server on local computer
    file_socket.connect((host, file_transferring_port))
    file_location = input("File Location of " + name + ": ")
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

    message = ""
    while True:
        option = input('1. Add a Expert\n 2.Evaluation on cluster\n 3. Exit\n')
        optionSelected = Switcher()
        message = optionSelected.indirect(option)

        if message == '1':
            s.send(message.encode('ascii'))
            file_sender("MOE")
            file_sender("TOE")
            file_sender("SOE")
            print("File Forwarded to the client")

        if message == '2':
            s.send(message.encode('ascii'))
            file_sender("MOE")

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
