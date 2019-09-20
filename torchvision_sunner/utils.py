from torchvision_sunner.constant import *

"""
    This script defines the function which are widely used in the whole package

    Author: SunnerLi
"""

def quiet():
    """
        Mute the information toward the whole log in the toolkit
    """
    global verbose
    verbose = False

def INFO(string = None):
    """
        Print the information with prefix

        Arg:    string  - The string you want to print
    """
    if verbose:
        if string:
            print("[ Torchvision_sunner ] %s" % (string))
        else:
            print("[ Torchvision_sunner ] " + '=' * 50)

def DEPRECATE(func_name, version):
    """
        Print the deprecated warning

        Arg:    func_name   (Str)   - The function or class name
                version     (Str)   - The version string
    """
    if verbose:
        print("[ Torchvision_sunner ] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("The function '{}' you used is deprecated in version {}, please use the alternative function in the future!!!!".format(func_name, version))
        print("[ Torchvision_sunner ] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")