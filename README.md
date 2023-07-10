# Machine Learning Overview

The notebook in this repository contains my Python machine learning notes. It is a work in progress; I update and improve my notes as I go along and use them as a reference for my projects.

Resources used are mainly notes from Oxford University ML courses and the books Deep Learning with Python by Francois Chollet and Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.


## AWS E2 Set-Up

1. Create VPC (Amazon Virtual Private Cloud):  
IPv4 CIDR gives range of IP addresses. Use https://cidr.xyz/. 10.0.0.0/16 gives 65,536 possible IP addresses. Spinning up a server will use one of them. This many gives a lot of room to grow. A dedicated VPC is unnecessary and costly, so we ignore.

<p align="center">
    <img src="readme_images\vpc.jpg" width="600"/>
</p>


2. Create Subnet:  
For a data centre, you set up firewalls; for the cloud, you set up subnets. Allows for connections to be limited/unlimited. Need to add an internet connection. As soon as I put any EC2 instances into this subnet, they will have access to the internet. Alternatively, we create a private subnet if we don't want access to the internet, e.g. for a MYSQL database.

A subnet is a smaller network of my resources. In my case, the resources are EC2 instances. Now need to create a smaller amount than default, ie 10.0.1.0/24 (256 IP addresses, 5 of which are reserved by AWS).

<p align="center">
    <img src="readme_images\subnet.jpg" width="600"/>
</p>

Need a public IP address. To do so, Enable auto-assign public IPv4 address in Edit subnet settings


<p align="center">
    <img src="readme_images\enable_ip.jpg" width="600"/>
</p>

Next, to EC2, select instance and create a keypair (or use existing key pair). Important for when we SSH into your server.

<p align="center">
    <img src="readme_images\keypair.jpg" width="600"/>
</p>


Seems like we're done but not quite. Running instance will make it forever try to connect but never be successful. Currently, our subnet has no public privilages (by default, no access to internet).

Add the Advanced Details bit from the link below in **`User Data`** (when creating EC2 instance):  
https://www.codingforentrepreneurs.com/blog/jupyter-notebook-server-aws-ec2-aws-vpc/  
It is a bash script to update the server, install essential builds, supervisor (so we can have different processes running), and other related services like git and nano. Also installs an additional firewall ufw (another layer of security). Change jupyter to jupyterlab.


```bash
#!/bin/bash

sudo apt-get update -y

sudo apt-get install build-essential libssl-dev libpq-dev libcurl4-gnutls-dev libexpat1-dev gettext unzip -y

sudo apt-get install supervisor -y 

sudo apt-get install python3-pip python3-dev python3-venv -y

sudo apt-get install nano -y

sudo apt-get install git -y 

sudo apt-get install nginx curl -y

sudo apt-get install ufw -y

sudo ufw allow 'Nginx Full'

sudo ufw allow ssh

sudo python3 -m pip install jupyterlab

sudo service supervisor start

sudo apt autoremove -y
```


Next, stick to 8 GiB (gp2) storage. Good enough for now.

Finally, add security group. Note, VPC is top level security. This is another layer below. Here we could add HTTP or HTTPS. We need SSH (secure shell) so we can log in.






```bash
# STEP 1: AWS set-up

# STEP 2: Add to Termius (screenshot of added...)
# Username: ubuntu



sudo apt update

mkdir ssl
cd ssl

openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

#conda config --add channels conda-forge

conda update --all -y

conda install jupyterlab
conda install nb_conda_kernels

conda create --name env_name python ipykernel
conda deactivate
conda install --name env_name tensorflow-gpu
conda update --name test --all


ipython -c "from notebook.auth import passwd; passwd()"
[copy output: 'argon2:$argon2id$v=19$m=10240,t=10,p=8$wXIvdKgQhn06i8bXKKjVnQ$7AUQm7sm0jroUghfyA0ZDoSd65sefBtjPuIx2GjjTDM']


jupyter notebook --generate-config
nano ~/.jupyter/jupyter_notebook_config.py



c = get_config()  #noqa

c.SeverApp.certfile = u'/home/ubuntu/ssl/cert.pem'
c.SeverApp.keyfile = u'/home/ubuntu/ssl/cert.key'
c.IPKernelApp.pylab = 'inline'
c.ServerApp.ip = '0.0.0.0'  #'*' works but not as safe? 
c.ServerApp.allow_root=True  #allow root access 
c.NotebookApp.open_browser = False
c.Sever.password = 'argon2:8$wXIvdKgQhn06i8bXKKjVnQ$7AUQm7sm0jroUghfyA0ZDoSd65sefBtjPuIx2GjjTDM'


jupyter lab

# In browser: (Public IPv4 address:port)
18.170.34.235:8888


```

### Optional

```bash
#OPTIONAL (SECURE THE KEY?)
chmod 400 JupyterKP.pem

#OPTIONAL (CONVERT KEY from ppk to pem)
sudo apt install putty-tools
puttygen ppk_file.ppk -O private-openssh -o pem_file.pem 


#USEFUL FOR FUTURE
sudo ssh -i JupyterKP-US.pem -L 8888:107.22.145.38:8888 ubuntu@ec2-107-22-145-38.compute-1.amazonaws.com

#SEE WHICH PORTS ARE IN USE:
sudo lsof -i -P -n | grep LISTEN
```



### Links
* https://www.youtube.com/watch?v=FEBYi8Ia8bk
* https://www.youtube.com/watch?v=TrqeGpOnqTU
* https://www.youtube.com/watch?v=qYe5J5lBvn4&t=446s