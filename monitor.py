import psutil
import signal
from psutil import process_iter
#from restart_ansys import restart 
import time
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
exec_loc = 'C:/Program Files/ANSYS Inc/ANSYS Student/v242/ansys/bin/winx64/ANSYS242.exe'


for proc in process_iter():
    for conns in proc.connections(kind='inet'):
        if conns.laddr.port == 8800: proc.send_signal(signal.SIGTERM) # or SIGKILL
mapdl = launch_mapdl(exec_loc, port = 8800, cleanup_on_exit = True)

j = 0
while True:
    flag = 0
    j += 1
    if j > 100:
        for proc in process_iter():
            with proc.oneshot():
                name = proc.name()
                if name == "ANSYS242.exe":
                    proc.send_signal(signal.SIGTERM) # or SIGKILL
                    print('auto kill')
    #        for conns in proc.connections(kind='inet'):
    #            if conns.laddr.port == 8800: proc.send_signal(signal.SIGTERM) # or SIGKILL
    #            print('auto kill')
        j = 0
        flag = 0

    for process in psutil.process_iter():
        with process.oneshot():
            name = process.name()
            if name == "ANSYS242.exe":
                flag = 1
    
    if flag == 0:
        i = 0
        while i <= 5:
            i += 1
            try:
                for proc in process_iter():
                    for conns in proc.connections(kind='inet'):
                        if conns.laddr.port == 8800: proc.send_signal(signal.SIGTERM) # or SIGKILL
                mapdl = launch_mapdl(exec_loc, port = 8800, cleanup_on_exit = True)
                break
            except:
                time.sleep(10)
    print('checked')
    time.sleep(15)
