import subprocess

path_code_python = "blender_event_data_generation.py"
#path_code_python = "__test.py"
subprocess.check_call("C:\\Program Files\\Blender Foundation\\Blender 3.5\\blender.exe --enable-event-simulate --python {} ".format(path_code_python))
