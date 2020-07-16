####Imports####
from numpy import asarray
from os import mkdir
from os.path import split, isdir
from webbrowser import open as open_webpage
from boxsdk import OAuth2
from http.server import BaseHTTPRequestHandler,HTTPServer
from boxsdk import Client
from datetime import date
import time


####Defining Global Constnant####
token_storage_file_name = 'tokens.txt'


#used in the box authorization
####Defining Server Object####
auth_code = None
#this is the basis of a http server that will be used to catch the access token
#for the authorization of the box client
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

#defining the http server object. It is set to run on the local address and
    #port: 0.0.0.0:8000
server = HTTPServer(('0.0.0.0', 8000), myHandler)




####Defining Box Folder Parser object####
class box_folder_parser:
    '''
    This object provieds the ability to parse through the contents of a box
    users' file storage aswell as the ability to download files from their box
    account to local storage.
    '''
    #initializing
    def __init__(self,client_id,client_secret,root_folder_id = '0'):
        '''
        parameters:
        client_id: str
            paremeter from box developer app
        client_secret: str
            paremeter from box developer app

        the above perameters come from box developers app. These tie the actions
        of the stcript to a specific box user. see the help documentation for
        how to set up the box app properlly such that this script will function
        correctly

        root_folder_id: str
            the folder id number for the desired root directory. This is used to
            change what the parser sees as the root directory. See the
            documentation for where to find the id number for a folder in box

        '''
        ####initialization####
        self.client_id = client_id
        self.client_secret = client_secret
        self.root_folder_id = root_folder_id
        #the current working directory. allways starts in what is defiend
            #as the root
        self.cwd_id = root_folder_id
    #authorizing and creating the box client
    def authorize_client(self):
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
                client_id = self.client_id,
                client_secret = self.client_secret,
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
                client_id = self.client_id,
                client_secret = self.client_secret,
            )

            #the url that will be opened so the the user can grant the
                #application authurization
            auth_url = 'https://account.box.com/api/oauth2/authorize?client_id='\
            + self.client_id + '&redirect_uri=' + \
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
    #returns a list of dict objects for the folders in the cwd
    def lsdir(self,l = True):
        '''
        l: Boolean
            whether or not to return in a numpy array format or a dictionary format
        returns:
            [dict{name,id}]
        desc:
            parses through the contents of the current working directory and
            returns information on the folders in the cwd
        return:
            a collection of folder information in either an array or dictionary
            format.
            default is array format
        '''
        #getting the client from the autherization
        #parsing the contents of the cwd
        items = self.client.folder(folder_id = str(self.cwd_id)).get_items()
        if l == True:
            #making a generater object that will gather the names an id numbers
                #of the foleders in the cwd
            names,ids = zip(*[[str(item.name),str(item.id)] for item in items\
                        if item._item_type == 'folder'])
            sub_folders = asarray([names,ids])
        else:
            #sorting the contents of the folder and pulling out only sub folders
            sub_folders = []
            for item in items:
                if str(item._item_type) == 'folder':
                    sub_folders.append({'name':item.name,'id':item.id})

        return(sub_folders)
    #'move' to a dif directory
    def chdir(self,id = None,folder_name = None):
        '''
        folder_name: string
            name of the folder to change directorys too
        id: string/int
            id number of the folder to change dir too

        operates like os.chdir in that it will select either a folder in the cwd
        that matches the name provide. in the event that an id number is
        supplied instead it will change to that folder directly. Currently
        this can only move to forlders in the cwd by name. by id you can select
        any folder to be the new cwd.
        '''

        #if there is just an id given
        if folder_name == None:
            if id == None:
                raise Exception('Neither an id number of folder name was given; cant chdir')
            self.cwd_id = str(id)
        #if a name is given
        else:
            #pulling all the sub directories
            sub_dirs = self.lsdir()
            #pulling out the id value coresponding to the name
            for dir in sub_dirs:
                if dir['name'] == folder_name:
                    self.cwd_id = str(dir['id'])
                    break
    #returns a list of dict objects for the files in the cwd
    def get_files(self,l = True):
        '''
        l: Boolean
            whether or not to return in a numpy array format or a dictionary format
        returns:
            [dict{name,id}]
        desc:
            parses through the contents of the current working directory and
            returns information on the files in the cwd
        return:
            a collection of file information in either an array or dictionary
            format.
            default is array format
        '''
        #getting the client from the autherization
        #parsing the contents of the cwd
        items = self.client.folder(folder_id = str(self.cwd_id)).get_items()
        #sorting the contents of the folder and pulling out only sub folders
        if l == True:
            #making a generater object that will gather the names an id numbers
                #of the files in the cwd
            names,ids = zip(*[[str(item.name),str(item.id)] for item in items\
                        if item._item_type == 'file'])
            files = asarray([names,ids])
        else:
            #generating other option: dict format
            files = []
            for item in items:
                if str(item._item_type) == 'file':
                    files.append({'name':str(item.name),'id':str(item.id)})

        return(files)
    #returns to the root directory
    def return_to_root(self):
        self.cwd_id = '0'
    #downloads a file bassed on its file id
    def download_file(self,file_id,save_location):
        '''
        file_id: string
            the id of the file that is to be downloaded
        save_location:
            the path to where the file should be saved

        Downloadeds a file, bassed on its id number, from box to local storage.
        '''
        try:
        #when the save location dosent exsist
            file_id = str(file_id)
            save_path = split(save_location)[0]
            if isdir(save_path) == False:
                mkdir(save_path)

            #selecting the file via id number.
            file = self.client.file(file_id)

            print('\tdownloading: ' + split(save_location)[1])

            #try to download the file 5 times
            n = 0
            while n < 5:
                try:
                    #opening a data stream to local storage
                    with open(save_location,'wb') as location:
                        #donwload the file from box to local
                        file.download_to(location)
                    #If the download was sucsefull, leave the loop
                    n = 5

                #when box has a problem for some unknown reason
                except urllib3.exceptions.DecodeError:
                    #dont do anything, just loop around and try again
                    n += 1
                    #if this is the last time the loop will happen
                    if n == 5:
                        raise Exception("Failed to download file: "\
                                                "urllib3.exceptions.DecodeError")
        except KeyboardInterrupt:
            time.sleep(1)


    def get_file_size(self,file_id):
        '''
        file_id:string
            The box file id to get the size of

        Returns the file size of a box file coresponding to file_id in bytes
        '''
        file = self.client.file(file_id)
        return(file.get().size)
