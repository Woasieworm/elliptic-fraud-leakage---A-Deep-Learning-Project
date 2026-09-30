import os
import platform
import subprocess
import sys


def setup_project():
    """Sets up the project virtual environment, installs dependencies safely

    inside the virtual environment, and prepares the .env file.
    """
    current_os = platform.system()
    print(f" Detecting operating system: {current_os}...")

    # 1. Determine venv name and its explicit internal Python path
    if current_os == "Windows":
        venv_name = "myenv"
        # Absolute or relative direct path to the environment's Python executable
        venv_python = os.path.abspath(os.path.join(venv_name, "Scripts", "python.exe"))
    else:  # macOS and Linux
        venv_name = ".venv"
        # Absolute or relative direct path to the environment's Python executable
        venv_python = os.path.abspath(os.path.join(venv_name, "bin", "python"))

    # 2. Create the Virtual Environment
    if not os.path.exists(venv_name):
        print(f" Creating virtual environment '{venv_name}'...")
        try:
            # Uses your system python to generate the isolated environment
            subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
            print(" Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f" Failed to create virtual environment: {e}")
            return
    else:
        print(f" Virtual environment '{venv_name}' already exists. Skipping.")

    # 3. Install Dependencies strictly inside the created venv
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print(f" Installing dependencies into '{venv_name}' using {venv_python}...")
        try:
            # Upgrade pip inside the virtual environment first to be safe
            subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # This explicitly forces installation inside the environment's site-packages
            subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file], check=True)
            print(f" Dependencies successfully isolated inside '{venv_name}'.")
        except subprocess.CalledProcessError as e:
            print(f" Failed to install dependencies inside venv: {e}")
            return
    else:
        print(f" '{requirements_file}' not found. Skipping installation.")

    # 4. Handle Environment Variables (.env setup)
    env_file = ".env"
    example_env = ".envexample"

    if os.path.exists(env_file):
        print("'.env' file already exists. Skipping creation.")
    elif os.path.exists(example_env):
        print(f"Creating '{env_file}' from '{example_env}'...")
        try:
            with open(example_env, "r") as src, open(env_file, "w") as dst:
                dst.write(src.read())
            print(" '.env' file created successfully.")
            print("Please open '.env' and add your Kaggle API token values.")
        except IOError as e:
            print(f" Failed to handle environment files: {e}")
    else:
        print(f" '{example_env}' not found. Cannot automatically build '.env'.")

    print("\n Setup complete!")
    if current_os == "Windows":
        print(f" IMPORTANT: To run code manually, activate first using: {venv_name}\\Scripts\\activate")
    else:
        print(f" IMPORTANT: To run code manually, activate first using: source {venv_name}/bin/activate")


if __name__ == "__main__":
    setup_project()
