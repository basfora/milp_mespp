import setuptools

with open("README.md", "r") as fh:
    my_long_description = fh.read()

setuptools.setup(name='milp_mespp',
                 version='0.0.1',
                 url='https://github.com/basfora/milp_mespp.git',
                 author='Beatriz A. Asfora',
                 author_email='beatriz.asfora@gmail.com',
                 packages=setuptools.find_packages(),
                 package_data={'milp_mespp': ['graphs/*.p']},
                 description="MILP models for MESPP problem",
                 long_description=my_long_description,
                 long_description_content_type="text/markdown",
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 install_requires=['gurobipy~=9.0.2',
                                   'matplotlib~=3.2.1',
                                   'numpy~=1.18.4',
                                   'scipy~=1.5.0',
                                   'pytest~=5.4.3',
                                   'setuptools~=39.0.1',
                                   ],
                 python_requires='>=3.6',
                 )
