import subprocess
import time
import os
import signal
import sys
import platform

IS_WINDOWS = platform.system() == "Windows"

def run_command(command, name):
    """Runs a shell command as a subprocess, with OS-specific termination handling."""
    print(f"\n--- Starting {name} ---")
    
    if IS_WINDOWS:
        process = subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        
    print(f"{name} started with PID: {process.pid}")
    return process

def terminate_process(process):
    """Terminates a subprocess gracefully, then forcefully if needed."""
    if process.poll() is None:
        print(f"Attempting to terminate process with PID: {process.pid}")
        if IS_WINDOWS:
            subprocess.run(f"TASKKILL /PID {process.pid} /T /F", shell=True, capture_output=True)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        try:
            process.wait(timeout=10)
            print(f"Process {process.pid} terminated gracefully.")
        except subprocess.TimeoutExpired:
            print(f"Process {process.pid} did not terminate gracefully, forcing kill.")
            if IS_WINDOWS:
                subprocess.run(f"TASKKILL /PID {process.pid} /T /F", shell=True, capture_output=True)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            print(f"Force-killed process with PID: {process.pid}")
    else:
        print(f"Process with PID: {process.pid} already terminated.")


def main():
    processes = []

    try:
        print("Starting ADAS MLOps Pipeline...")

        print("\n--- Starting Data Generator (data_simulator.py) ---")
        data_gen_process = run_command("python data_simulator.py", "Data Generator")
        processes.append(data_gen_process)

        fastapi_command = "uvicorn pipeline.model_deployment:app --host 0.0.0.0 --port 8000 --reload"
        fastapi_process = run_command(fastapi_command, "FastAPI App (app.py)")
        processes.append(fastapi_process)
        time.sleep(5)

        mlops_orch_process = run_command("python pipeline/mlops_orchestrator.py", "MLOps Orchestrator (mlops_orchestrator.py)")
        processes.append(mlops_orch_process)

        print("\nAll pipeline components are running. Press Ctrl+C to stop them all.")
        print("You can access the FastAPI docs at: http://127.0.0.1:8000/docs")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating all pipeline components...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        for p in processes:
            terminate_process(p)
        print("All pipeline components have been stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()