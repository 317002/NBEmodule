'''
Module for  basic interaction between a box account and the user's client


What the module can do:

1 The ability to ask a user for read/wright permission for their box account
2 The ability to download files bassed on some file structure path
  -Also the ability to downlad files bassed on the their unique box id number
3 The ability to parse through the file structure of a box file system
4 The ability to determin the box-ID of a file/folder bassed on a file path
'''
from webbrowser import open as open_webpage
from boxsdk import OAuth2
from http.server import BaseHTTPRequestHandler,HTTPServer
from boxsdk import Client
from boxsdk.object.item import Item as boxItem
from collections import namedtuple
from urllib3.exceptions import DecodeError

auth_code = None

class myHandler(BaseHTTPRequestHandler):
    '''
    Basic http server intented to be hosted on a local address

    listens for a box authorization code. Box will redirect the user to the
    address explained in the guide(the same one this server should be hosted on)
    and in the porcesse give an auth code used in client -> box comunication
    '''
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
        '''
        :param clientID:
            Id string connecting the client's actions to a user's box app
        :type clientID:
            string
        :param clientSecret:
            secret id string connecting the client's actions to a user's box app
        :type clientSecret:
            string
        '''

        self.clientID = clientID
        self.clientSecret = clientSecret

        #requesting read-wright permission of user's box account
        self.authorize_client()

    def authorize_client(self):

        '''Methode for giving client read/wright permission for users box account
        '''

        token_storage_file_name = './tokens.txt'

        #checking for prior authorization
        try:
            #reading the accesse token and the refresh token from a previous
            #sesions if it exsists
            with open(token_storage_file_name,'r') as file:
                access_token,refresh_token = \
                            [c for c in file.read().split('\n') if c != '\n']

            #initializing the authorization object from boxsdk
            oauth = OAuth2(
                client_id = self.clientID,
                client_secret = self.clientSecret,
                access_token = access_token,
                refresh_token = refresh_token,
            )

            #initializing the client
            self.client = Client(oauth)

            ####checking if the tokens are still valid####
            #we try to get the properties of the root folder. If this fails
                #it probably means that tokens have expired. When the tokens
                #expire
            folder = self.client.folder(folder_id = '0')
            folder.get()
            del folder


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
            access_token, refresh_token =\
                                            oauth.authenticate(auth_code)

            #saving the access_token and refresh_token so the user wont be
                #prompted to authorize again for a periode of time.
            with open(token_storage_file_name,'w') as file:
                file.write(access_token + '\n' + refresh_token)


            self.client = Client(oauth) #initializing the box client

            #checks that authorization was sucsefully
            try:
                folder = self.client.folder(folder_id = '0')
                folder.get()
                del folder
            except:
                raise Exception('Failed to authorize client')
                del folder

        #clean up
        try:
            del folder
        except NameError:
            pass

    def boxFile(self,id):
        '''Returns a box file obj from a given id

        :param id:
            The id string for the file on box to be repersented
        :type id:
            string
        :return:
            box file obj coresponding to the id string
        :rtype:
            :class:'boxsdk.object.file.File'
        '''
        return self.client.file(id)

    def boxFolder(self,id):
        '''Returns a box Folder bassed on the id number provided

        :param id:
            id string coresponding to a folder on box
        :type id:
            string
        :return:
            box folder obj coresponding to the id given
        :rtype:
            :class:'boxsdk.object.folder.Folder'
        '''
        return self.client.folder(id)

    def getParentFolder(self,boxObject = None):
        '''Gets the name and id of the parrent folder for the object provided

        :parma boxObject:
            box object from boxsdk module
        :type boxObject:
            :class:'boxsdk.object.folder.Folder' #for exsample
        :param return:
            box folder object for the parrent directory
        :rtype:
            :class:'boxsdk.object.folder.Folder'
        '''
        parentInfo = boxObject.get().parent


        if parentInfo != None:
            parrentId = parentInfo.id
        elif id != '0':#means the folder is in the root dir
            parrentId = '0'
        else:
            parrentId = None



        return self.client.folder(parrentId)

    def box_lsdir(self,parrent):
        '''returns a box items iterator object coresponding to the contents of
        parrent folder

        :param parrent:
            Box folder that we are trying to observer the contents of
        :type parrent:
            :class:'boxsdk.object.folder.Folder'
        :return:
            iterator of box items in the parrent folder
        :rtype:
            :class:'boxsdk.pagination.limit_offset_based_object_collection.LimitOffsetBasedObjectCollection'
        '''
        return parrent.get_items()

    def download(self,item,saveLocation):
        '''Downloads the file repersented by the box file object

        :param item:
            box file object coresponding to the file to be downloaded
        :type item:
            :class:'boxsdk.object.file.File'
        :param saveLocation:
            Where to save the file, including the name and file exstention

            The name+ext can be gathered from item.name
        :type saveLocation:
            string
        '''

        #trys to download the file 5 times
        for n in range(5):
            try:
                #download the file
                with open(saveLocation,'wb') as location:
                    item.download_to(location)
                break
                #when there is an issue downloading the file
            except DecodeError:
                pass
        else:
            raise Exception('Failed to download: ' + saveLocation.split('/')[-1])
