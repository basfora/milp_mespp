import os

this_path = os.path.dirname((os.path.abspath(__file__)))
my_command = """echo "export PYTHONPATH="${PYTHONPATH}:/home/beatriz/PyCharmProjects"" >> ~/.bashrc"""
up_file = """source ~/.bashrc"""

os.system(my_command)
os.system(up_file)
