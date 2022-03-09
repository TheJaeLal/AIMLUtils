import os
from pathlib import Path
import shutil


def new_suffix(files, suffix):
    return list(map(lambda file: str(Path(file).with_suffix(suffix)), files))


def filecount(dir_path, glob_pattern="*"):
    return len(list(Path(dir_path).glob(glob_pattern)))
    
    
def exists(file_paths, verbose=False):
    """
        file_paths : str/Path -> Returns bool
        file_paths: Collection of str/Path -> Returns dict of str/path:bool
    """
    
    def single_file_exists(file_path, verbose):
        file_exists = os.path.exists(str(file_path))
        if not file_exists:
            print(f"***Path: {file_path} does not exist***")
        return file_exists
        
    if isinstance(file_paths, (str, Path)):
        return single_file_exists(file_paths, verbose)
    
    status = {}
    # Assuming collection of strings/paths
    for path in file_path:
        status[path] = single_file_exists(path, verbose)
        
    return status
        

#TODO: Convert this into a partial of _transfer_all
def copy_all(filenames, src_dir, dst_dir, warnings=True):
    _transfer_all(filenames, src_dir, dst_dir, copy=True, warnings=warnings)

    
def move_all(filenames, src_dir, dst_dir, warnings=True):
    _transfer_all(filenames, src_dir, dst_dir, copy=False, warnings=warnings)

    
def _transfer_all(filenames, src_dir, dst_dir, copy, warnings=False):
    """
        #TODO: Docstring
    """
    for _dir in (src_dir, dst_dir):        
        if not exists(_dir, verbose=warnings):
            return
        
    for file_name in filenames:
        
        # To handle str as well as pathlib.Path 
        file_name = Path(file_name).name

        src_path = Path(src_dir) / file_name
        dst_path = Path(dst_dir) / file_name

        # Check to ensure src-file exists        
        if exists(src_path, verbose=warnings):
            if copy:
                shutil.copy(str(src_path), str(dst_path))
            else:
                shutil.move(str(src_path), str(dst_path))

    return


def delete_all(filenames, _dir, dryrun=True):
    
    """
        #TODO: Docstring
    """
    
    if dryrun:
        print("+++Doing a dry run, to actually delete files, pass dryrun=False+++")
        
    if not exists(_dir, verbose=True):
        return
        
    for file_name in filenames:
        
        # To handle str as well as pathlib.Path 
        file_name = Path(file_name).name
        
        file_path = Path(_dir) / file_name
        
        if exists(file_path, verbose=True):            
            if dryrun:
                print("**Deleting:", str(file_path))
            else:
                os.remove(str(file_path))
            
    return
    
