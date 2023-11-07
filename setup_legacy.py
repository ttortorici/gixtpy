import sys
import subprocess

packages_to_install = ['pip',
                       'numpy',
                       'scipy',
                       'matplotlib',
                       'tk',
                       'tkinter',
                       'tifffile']

if __name__ == "__main__":
    for package_name in packages_to_install:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', package_name])
