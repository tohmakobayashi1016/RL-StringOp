import subprocess
import os
import time 
import win32com.client

#rhino command: python open_rhino.py 
def open_rhino():
    # Path to Rhino7 executable
    rhino_path = r"C:\Program Files\Rhino 7\System\Rhino.exe"
    
    # Open Rhino7 if it exists and no instance is running
    if os.path.exists(rhino_path):
        subprocess.Popen([rhino_path])
        print("Rhino 7 is opening...")
    else:
        print("Rhino 7 executable not found at the specified path.")
        return None
        
    
#grasshopper command
def open_grasshopper_and_load_file(gh_file_path):
    try: 
        #Connect to the running Rhino instance
        rhino = win32com.client.Dispatch("Rhino.Application")
        rhino.Visible = True
        
        rhino.RunScript("_Grasshopper", 0)
        print("Launching Grasshopper...")
        
        time.sleep(5)
        
        rhino.RunScript(f'_GrasshopperOpen "{gh_file_path}"',0)
        print(f"Grasshopper file {os.path.basename(gh_file_path)} is opening...")

    except Exception as e:
        print(f"An error occurred: {e}")
    

if __name__ == "__main__":
    start_time = time.time()

    open_rhino()

    #Wait for Rhino to fully open before running the next part
    time.sleep(10)
    
    #Note: Application closes after script execution since script runs to completion
    gh_file_path = r"C:\Users\footb\Desktop\Thesis\String-RL\GH files\activation_check.gh"
    open_grasshopper_and_load_file(gh_file_path)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script execution time: {elapsed_time: .2f} seconds")