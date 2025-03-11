from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from zipfile import ZipFile
import os
from tqdm import tqdm
import tempfile

# refer to pydrive2 documentation for more information
# you need to setup Google Drive API and OAuth 2.0 client ID first

gauth = GoogleAuth()
# gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

PARENT_ID = '' # insert file id

# upload hello.txt to Google Drive
# file = drive.CreateFile({
#     'parents': [{'id': PARENT_ID}],
#     'title': 'hello.txt',
#     'mimeType': 'text/plain'
# })
# file.SetContentString('Hello World!')
# file.Upload()

def traverse_path(dir_parts):
    current_id = PARENT_ID
    for dir in dir_parts:
        folder = None
        file_list = drive.ListFile({
            'q': f"'{current_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder' and title='{dir}'"
        }).GetList()
        if len(file_list) > 0:
            folder = file_list[0]
        if folder is None:
            folder = create_folder(current_id, dir)
        current_id = folder['id']
    return current_id

def create_folder(parent_id, folder_name):
    folder = drive.CreateFile({
        'parents': [{'id': parent_id}],
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    })
    folder.Upload()
    return folder

with ZipFile('../dataset/tvb-hksl-news-v1.zip') as zf:
    for file_path in tqdm(zf.namelist()):
        path_parts = file_path.split('/')
        dir_parts = path_parts[:-1]
        file_name = path_parts[-1]
        # if the file_name does not contain a '.', it is a folder
        if '.' not in file_name:
            continue
        # for every path part, recursively create a folder in Google Drive if it doesn't exist
        current_id = traverse_path(dir_parts)
        # upload the file
        file = drive.CreateFile({
            'parents': [{'id': current_id}],
            'title': path_parts[-1]
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            # read the file from the zip file first and print its contents in console
            zf.extract(file_path, temp_dir)
            file.SetContentFile(os.path.join(temp_dir, file_path))
            file.Upload()
            file = None
