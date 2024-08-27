import subprocess
import sys

def start_server():
    subprocess.Popen(["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db",
                      "--default-artifact-root", "./mlruns", "--host", "0.0.0.0", "--port", "5001"])

def stop_server():
    subprocess.run(["pkill", "-f", "mlflow server"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            start_server()
        elif sys.argv[1] == "stop":
            stop_server()