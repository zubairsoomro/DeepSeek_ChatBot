# DeepSeek_ChatBot
Development of an Offline Voice-Enabled ChatBot with Document Processing Capabilities Using DeepSeek-R1 Algorithm

# DeepSeek
A Comprehensive Repository for my Experiments with DeepSeek. 

# Python Setup

There are several ways you can install Python and set up your computing environment.

I am using this project code running windows.


## 1. Download and install Conda

Download Anaconda on your system and create Virtual Enviroment for this project.

Link: https://www.anaconda.com/download | 

After downloading it, Copy the path where anaconda installed and add it to your system enviroment varibale 

## 2. Create a new virtual environment

After the installation was successfully completed, I recommend creating a new virtual environment called `LLMs`, which you can do by executing

Open Anaconda and write below command.

```bash
conda create -n LLMs python=3.12
```

> Many scientific computing libraries do not immediately support the newest version of Python. Therefore, when installing PyTorch, it's advisable to use a version of Python that is one or two releases older. For instance, if the latest version of Python is 3.13, using Python 3.11 or 3.12 is recommended.

Next, activate your new virtual environment (you have to do it every time you open a new terminal window or tab):

```bash
conda activate LLMs
```
## 3. Install PyTorch

PyTorch can be installed just like any other Python library or package using pip. For example:

```bash
pip install torch
```

However, since PyTorch is a comprehensive library featuring CPU- and GPU-compatible codes, the installation may require additional settings and explanation.

It's also highly recommended to consult the installation guide menu on the official PyTorch website at [https://pytorch.org](https://pytorch.org).

<img src="https://raw.githubusercontent.com/Sangwan70/Building-an-LLM-From-Scratch/refs/heads/main/setup/images/pytorch-installer.webp" width="600px">


## 4: Downloading and Installing Ollama

Ollama is an open-source framework designed for running, managing, and interacting with large language models (LLMs) locally on your machine. It simplifies downloading, running, and customizing AI models without needing cloud-based services, In Our case, it will connect you to the DeepSeek-r1 lattest model.
[https://ollama.com/download****](https://ollama.com/download)

After installation, run this command on your VS Code termal or according your project framework terminal. 
ollama run deepseek-r1


## 4. Installing Python packages and libraries used in this course

Please refer to the requirement.txt 

## 5 Main file execution 

After installing all above requirements. 

Run main file on your terminal
streamlit run app.py 
Open another terminal and run this command
ollama run deepseek-r1

Any questions? Please feel free to reach out on linkedIn.
https://www.linkedin.com/in/zubair-soomro-bb4699153




