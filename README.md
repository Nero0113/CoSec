# egd4security_hardening



## ✨Overview

![framework of our method](/figure/frame7.png)

Code for our paper "On-the-Fly Security Hardening of Code LLMs via Supervised Co-Decoding"

## Directory Structure

The directory structure of this repository is shown as below:

```
.
|-- data_train_val    # Dataset for training and validation 
|-- data_eval          # Dataset for evaluation
|-- results	            # Experimental results
|-- scripts             # Scripts for training and inference
|-- trained	           # Trained LoRA for security.
|-- transformers	   # The source code of transformers version 4.33.0 obtained from the
                           # official website, as well as our proposed security hardening
                           # generation framework.

```

We used the same training and validation data as SVEN. 

[SVEN]: https://github.com/eth-sri/sven

You can get our security hardening code here

```
.
|-- transformers
   |-- generation
      |-- utils                   # You can get our security hardening code here
```

## 🔨 Setup

```
conda create -n code_sec python==3.8
conda activate code_sec
pip install -r requirements.txt
```

## 🚀 Train

To train a LoRA for security yourself, run:

```
$ python train_lora_sec.py
```

