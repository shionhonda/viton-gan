import os 

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)