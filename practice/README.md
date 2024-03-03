# Instruction to set up the course Conda virtual environment 

We will use Python throughout the course. For practice session and homework, we will manage all library packages under a Conda virtual environment. As a first step, please install the Python 3 version of Anaconda (https://www.anaconda.com/download). It will provide a cross-platform installation package. 

*** Windows Users: please use Linux running on Windows such as Ubuntu on Windows (https://canonical-ubuntu-wsl.readthedocs-hosted.com/en/latest/) and then run Python on it ***


1. We will use the `FMP' conda environment. Please, go to the FMP website and follow the instruction at this link (https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_GetStarted.html)


2. (Tip #1) If you encounter the error message 'No module named pip', type the following commands:

```
    conda activate FMP (activate the FMP environment)  
    python -m ensurepip (install pip in the FMP environment)  
    conda env update -f environment.yml (continue to install the FMP environment)  
```

3. (Tip #2) If you encounter the error message '/usr/lib/liblapack.3.dylib' (no such file, not in dyld cache), type the following commands:

```
    conda install -c conda-forge librosa (install Librosa)
    conda install lapack 
    conda install scipy
```

4. Run Jupyter Notebook 

```
   jupyter notebook 	
```

5. To deactivate the active environment, use
    
```
   conda deactivate
```

