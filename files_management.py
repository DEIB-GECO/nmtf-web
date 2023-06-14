from heapq import heapify, heappush, heappop
from time import time
import threading
import shutil
import string
import random
import os
from werkzeug.utils import secure_filename

ROUTINE_S = 30  # 2 minutes
FILE_LIFE_S = 60 * 1  # 20 minutes
UNIQUE_NOT_USED_LIFE = 30  # 1 minute
ROUTINE_UNIQUE_NOT_USED = 30  # 30 seconds
Uniques = set()  # HashMap containing unique dir names
Heap = []  # min heap that contains unique dir names and the time at they were created
# set up heap
heapify(Heap)

uniques_not_used = dict()  # dictionary that contains unique dir names not already used and time at they were created
uniques_requests = set()  # set that contains uniques that have a request elaboration in progress

SECRET_ADD = "_dir/"  # safety string added at every unique dir name
DIR_RESULTS = "results/"
FN_RESULTS = "results.txt"  # file name where results are saved
IMG_EXTENSION = "png"
UPLOAD_FOLDER = "files/"  # where user's dir and files are created
SETUP_FILE = "graph_topology."
A_EXTENSION = {"txt"}  # association file extension
S_EXTENSION = {"yaml"}  # setting file extension
L = 20  # uniques length

uniques_lock = threading.Lock()
uniques_not_used_lock = threading.Lock()
heap_lock = threading.Lock()
requests_lock = threading.Lock()


def check_request(unique):
    """
    Check if unique as already associated to a task

    Parameters
    ----------
    unique : str
        unique string

    Returns
    -------
    bool
        True if the unique can be associated to another task, False otherwise
    """
    with requests_lock:
        res = unique in uniques_requests
        if not res:
            uniques_requests.add(unique)
    return not res


#
#  Name: get_path
#  Parameters: (unique)
#  Return: path:String
#  Global variables: UPLOAD_FOLDER, SECRET_ADD
#  Description: return path where files are saved
#
#  UPLOAD_FOLDER + unique + SECRET_ADD
#  The dir + the check of just alphanumerical chars is needed so for sure
#  no one can access other file.
#  The final "_dir" it's as a secret password so even if a third
#  person knows the unique name and wants to access directly to the files
#  he can not
#
def get_path(unique):
    """
    Get the path where are saved files associated with a specific unique

    Parameters
    ----------
    unique : str
        unique string

    Returns
    -------
    path : str
        path = UPLOAD_FOLDER + unique + SECRET_ADD
    """
    path = UPLOAD_FOLDER + unique + SECRET_ADD
    return path


def get_standard_setting_file_name():
    """
    Get the standard name for the setting file

    Returns
    -------
    str
        name of the standard file (.yaml)
    """
    return SETUP_FILE+"yaml"


def get_setting_file_name(file):
    """
    Create file name for the setting file, preserving its extension

    Parameters
    ----------
    file : FileStorage
        setting file

    Returns
    -------
    str
        SETUP_FILE + extension
    """
    return SETUP_FILE + file.filename.rsplit('.')[1]


#
#  Name: get_path_results
#  Parameters: (unique)
#  Return: path_res:String
#  Global variables: UPLOAD_FOLDER, SECRET_ADD
#  Description: return path where file with results is saved
#
def get_path_results(unique):
    """
    Get the path where file with results is saved

    Parameters
    ----------
    unique : str
        unique string

    Returns
    -------
    str
        path of result file
    """
    path_res = UPLOAD_FOLDER + unique + SECRET_ADD + DIR_RESULTS + FN_RESULTS
    return path_res


# Node of the heap
class UniqueAndTime:
    """
    Node of a heap to keep track of the unique used. After a time of FILE_LIFE_S seconds, all files associated with this
    unique is eliminated.

    Attributes
    ----------
    unique : str
        Unique string
    time : float
        time at which this node was created

    Parameters
    ----------
    unique : str
        unique string
    """
    # constructor
    def __init__(self, unique):
        self.unique = unique
        self.time = time()

    # override the comparison operator
    def __lt__(self, nxt):
        return self.time < nxt.time


#
#  Name: routine_clean_heap
#  Parameters: ()
#  Return: void
#  Global variables: Heap, Uniques, ROUTINE_S
#  Description: Cleans Heap and Uniques hashmap every ROUTINE_S seconds
#
def routine_clean_heap():
    """
    Function to stat the routine to clean up uniques used.
    Called every ROUTINE_S seconds.

    See Also
    --------
    UniqueAndTime
    """
    # print("Heap in clean:")
    # for x in Heap:
    #    print(x.unique)
    print("Cleaning heap...")
    # print(Uniques)
    with heap_lock:
        try:
            while time() - Heap[0].time > FILE_LIFE_S:
                dir_name = heappop(Heap).unique
                # remove dir and unique
                with uniques_lock:
                    with requests_lock:
                        shutil.rmtree(get_path(dir_name))
                        Uniques.remove(dir_name)
                        uniques_requests.remove(dir_name)  # so unique
        except IndexError as e:
            # print("Index Error during heap cleaning", e, "Time:", time())
            pass  # Heap is empty
        except OSError as e:
            print("OS Error during heap cleaning", e, "Time:", time())
    threading.Timer(ROUTINE_S, routine_clean_heap).start()


def routine_clean_uniques_not_used():
    """
    Function to stat the routine to clean up uniques created but never used.
    Called every ROUTINE_UNIQUE_NOT_USED seconds.
    """
    print("Cleaning uniques_not_used...")
    with uniques_not_used_lock:
        keys = list(uniques_not_used.keys())
        for unique in keys:
            try:
                if time() - uniques_not_used[unique] > UNIQUE_NOT_USED_LIFE:
                    uniques_not_used.pop(unique)
                    with uniques_lock:
                        shutil.rmtree(get_path(unique))
                        Uniques.remove(unique)
                    print("Unique not used", unique, "was removed")
            except OSError as e:
                print("OS Error during uniques not used cleaning", e, "Time:", time())
    threading.Timer(ROUTINE_UNIQUE_NOT_USED, routine_clean_uniques_not_used).start()


#
#  Name: new_unique_name
#  Parameters: (len:Integer)
#  Return: fn:String
#  Global variables: Uniques, L
#  Description: Creates a new 'unique name'=fn of length=len
#
#  The 'unique name' is an alphanumerical string
#
def new_unique_name():
    """
    Create a new unique

    Returns
    -------
    str
        new unique
    """
    u = ''
    with uniques_lock:
        while True:
            u = u.join(random.choices(string.ascii_lowercase +
                                      string.ascii_uppercase + string.digits, k=L))
            if u not in Uniques:
                break
        Uniques.add(u)
    return u


#
#  Name: check_unique
#  Parameters: (unique)
#  Return: ok:Bool
#  Global variables: L, Uniques
#  Description: return true if unique is safe
#
def check_unique(unique):
    """
    Check if a unique exists

    Parameters
    ----------
    unique : str
        unique to check

    Returns
    -------
    bool
        True if the unique exists, False otherwise
    """
    with uniques_lock:
        ok = False
        if len(unique) == L and unique.isalnum() and unique in Uniques:
            ok = True
        return ok


#
#  Name: create_dir
#  Parameters: (unique)
#  Return: void
#  Description: creates dir from unique. DOES NOT ADD unique to Heap
#
def create_dir(unique):
    """
    Creates dir for unique. DOES NOT ADD unique to Heap

    Parameters
    ----------
    unique : str
        unique
    """
    os.mkdir(get_path(unique))
    os.mkdir(get_path(unique)+DIR_RESULTS)


#
#  Name: is_file_safe
#  Parameters: (file, is_setting)
#  Return: ok:Bool
#  Global variables: A_EXTENSION, S_EXTENSION
#  Description: Check if association (setting if is_setting == True) file is safe: extension and if name contains
#  just alphanumerical char or '_' and len less than L
#
def is_file_safe(file, is_setting):
    """
    Check if file is safe: extension and if name contains just alphanumerical char or '_' and len less than L.

    Parameters
    ----------
    file : FileStorage
        file to check
    is_setting : bool
        if False check file as association file, if True check file as setting file

    Returns
    -------
    bool
        True if the file is safe, False otherwise
    """
    ok = False
    filename = file.filename
    if '.' in filename:
        split = filename.rsplit('.')
        if len(split) == 2 and len(split[0]) <= 30:
            extensions = A_EXTENSION
            if is_setting:
                extensions = S_EXTENSION
            if split[1].lower() in extensions:
                ok = True
                for c in split[0]:
                    if not (c.isalnum() or c == "_"):
                        ok = False
                        break
    return ok


#
#  Name: check_config_file
#  Parameters: (file)
#  Return: ok:Bool
#  Description: Check if setting file is safe and well formatted
#
def check_config_file(file):
    """
    Check if setting file is safe and well formatted

    Parameters
    ----------
    file : FileStorage
        setting file

    Returns
    -------
    bool
        True if the file is safe, False otherwise
    """
    ok = True
    if not is_file_safe(file, True):
        return False
    # check file format
    return ok


#
#  Name: check_files
#  Parameters: (file)
#  Return: ok:Bool
#  Description: Check if association file is safe and well formatted
#
def check_files(files):
    """
    Check if association files are safe and well formatted

    Parameters
    ----------
    files : Union[List[FileStorage], list]
        association files

    Returns
    -------
    bool
        True if all files are safe, False otherwise
    """
    ok = True
    for file in files:
        if not is_file_safe(file, False):
            return False
    # check file format
    return ok


#
#  Name: save_file
#  Parameters: (file, unique:String)
#  Return: void
#  Description: Save file in unique dir (get_path(unique))
#
def save_file(file, unique, is_setting=False):
    """
    Save file in unique dir (get_path(unique))

    Parameters
    ----------
    file : Union[List[FileStorage], list]
        association files
    unique : str
        unique associated with the file
    is_setting: str, default=False
        Set to True if file is a setting file, set to False if file is an association file
    """
    if not is_setting:
        file.save(get_path(unique) + file.filename)
    else:
        file.save(get_path(unique) + get_setting_file_name(file))


#
#  Name: save_files
#  Parameters: (files, unique)
#  Return: void
#  Description: Save files in get_path(unique)
#
def save_files(files, unique):
    """
    Save files in unique dir (get_path(unique))

    Parameters
    ----------
    files : Union[List[FileStorage], list]
        association files
    unique : str
        unique associated with the files
    """
    for file in files:
        save_file(file, unique)


#
#  Name: add_to_heap
#  Parameters: (unique)
#  Return: void
#  Global variables: Heap
#  Description: add unique to Heap
#
def add_to_heap(unique):
    """
    Add unique to used uniques.

    Parameters
    ----------
    unique : str
        unique used

    See Also
    --------
    UniqueAndTime
    """
    with heap_lock:
        heappush(Heap, UniqueAndTime(unique))


def add_to_unique_not_used(unique):
    """
    Add unique to uniques created but never used.

    Parameters
    ----------
    unique : str
        unique never used
    """
    with uniques_not_used_lock:
        uniques_not_used[unique] = time()


def remove_from_unique_not_used(unique):
    """
    Unique was used or its lifetime is expired.
    Remove unique to uniques created but never used.

    Parameters
    ----------
    unique : str
        unique
    """
    with uniques_not_used_lock:
        try:
            uniques_not_used.pop(unique)
            return True
        except KeyError:
            return False


def get_images_list(unique):
    """
    Get all the images associated with a unique

    Parameters
    ----------
    unique : str
        unique

    Returns
    -------
    List[str]
        list of the images path
    """
    filelist = os.listdir(get_path(unique) + DIR_RESULTS)
    for file in filelist[:]:  # filelist[:] makes a copy of filelist.
        if not (file.endswith("."+IMG_EXTENSION)):
            filelist.remove(file)
    return filelist


def get_path_img_res(unique, img_name):
    """
    Get the path of a image associated with a unique

    Parameters
    ----------
    unique : str
        unique
    img_name : str
        name of the image with extension

    Returns
    -------
    str
        image path
    """
    split = img_name.rsplit('.')
    if len(split) == 2 and split[1] == IMG_EXTENSION:
        return get_path(unique) + DIR_RESULTS + secure_filename(img_name)
    raise AttributeError
