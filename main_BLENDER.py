import subprocess
import os
path_code_python = "blender_event_data_generation_3.py"
#path_code_python = "__test.py"
#subprocess.check_call("C:\\Program Files\\Blender Foundation\\Blender 3.5\\blender.exe --enable-event-simulate --python {} ".format(path_code_python))
#subprocess.check_call("./blender --enable-event-simulate --python {} ".format(path_code_python))
os.system("blender --enable-event-simulate --python {} ".format(path_code_python))
