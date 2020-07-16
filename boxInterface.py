'''
Module for  basic interaction between a box account and the user's client


What the module can do:

1 The ability to ask a user for read/wright permission for their box account
2 The ability to download files bassed on some file structure path
  -Also the ability to downlad files bassed on the their unique box id number
3 The ability to parse through the file structure of a box file system
4 The ability to determin the box-ID of a file/folder bassed on a file path
'''

from numpy import asarray
from os import mkdir
from os.path import split, isdir
from webbrowser import open as open_webpage
from boxsdk import OAuth2
from http.server import BaseHTTPRequestHandler,HTTPServer
from boxsdk import Client
from datetime import date
import time

auth_code = None

class myHandler(BaseHTTPRequestHandler):
	#Handler for the GET requests
    def do_GET(self):
        if '?code=' in self.path:
            global auth_code
            auth_code = self.path[7:]
            self.wfile.write(bytes("<html><head><title>access was granted(This tab can be closed).</title></head>", "utf-8"))
            #self.request.settimeout(60)
        elif 'access_denied' in self.path:
            self.wfile.write(bytes("<html><head><title>access was DENIED(This tab can be closed)</title></head>", "utf-8"))
            auth_code = 'denied'
        else:
            self.wfile.write(bytes("<html><head><title>'How did you get here?!?'</title></head>", "utf-8"))
            pass
        return


server = HTTPServer(('0.0.0.0', 8000), myHandler)

class boxInterface:

    def __init__(self,clientID,clientSecret):
        self.clientID = clientID
        self.clientSecret = clientSecret

    def authorize_client(self):
        token_storage_file_name = './tokens.txt'
        '''
        Will prompt for authorization to interact with the user's box files.
        '''
        #checking for prior authorization
        try:
            #reading the accesse token and the refresh token from a previous
            #sesions if it exsists
            with open(token_storage_file_name,'r') as file:
                self.access_token,self.refresh_token = \
                            [c for c in file.read().split('\n') if c != '\n']

            #initializing the authorization object from boxsdk
            oauth = OAuth2(
                client_id = self.clientID,
                client_secret = self.clientSecret,
                access_token = self.access_token,
                refresh_token = self.refresh_token,
            )

            #initializing the client
            self.client = Client(oauth)

            ####checking if the tokens are still valid####
            #we try to get the properties of the root folder. If this fails
                #it probably means that tokens have expired. When the tokens
                #expire
            folder = self.client.folder(folder_id = '0')
            folder.get()


        #for one reason or another authorization hast to be granted again
        #mostlikely becuase the authorizatioin token has expired
        except:
            #partially initializing the authorization object from boxsdk
            oauth = OAuth2(
                client_id = self.clientID,
                client_secret = self.clientSecret,
            )

            #the url that will be opened so the the user can grant the
                #application authurization
            auth_url = 'https://account.box.com/api/oauth2/authorize?client_id='\
            + self.clientID + '&redirect_uri=' + \
            'http://0.0.0.0:8000' + '&response_type=code'

            #opening authorization url
            open_webpage(auth_url)

            #Listing on the server for the authcode
            #looking for a change in the global auth_code variable
            while auth_code == None:
                #loop breaks once the auth code is grabbed.
                #the server closes once the code is sucsefully obtained as
                    #a result
                server.handle_request()
            if auth_code == 'denied':
                raise Exception('Authorizatioin was denied.')


            
            #fully initializing the authorization object from boxsdk
            self.access_token, self.refresh_token =\
                                                oauth.authenticate(auth_code)






            #saving the access_token and refresh_token so the user wont be
                #prompted to authorize again for a periode of time.
            with open(token_storage_file_name,'w') as file:
                file.write(self.access_token + '\n' + self.refresh_token)



            #initializing the client
            self.client = Client(oauth)
            try:
                folder = self.client.folder(folder_id = '0')
                folder.get()
            except:
                raise Exception('Failed to authorize client')


'''!!!!!!MAKE SURE TO REMOVE THIS!!!!!!'''
    #this being the client id and secret
#values that tie the actions of the script to that of a singal account
clientId = 'wlvj07x8beuuoehko90152d7j0331p6m'
clientSecret = 'Eq2N2g5Go6h3waxfkTXwqoLVor7QgjqI'

def main():
    box = boxInterface(clientId,clientSecret)
    box.authorize_client()


if __name__ == '__main__':
    main()
