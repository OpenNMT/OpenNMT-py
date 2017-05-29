## Standard

1\. [Install Torch](http://torch.ch/docs/getting-started.html)

2\. Install additional dependencies:

```bash
luarocks install tds
```

3\. Clone the OpenNMT repository:

```bash
git clone https://github.com/OpenNMT/OpenNMT
cd OpenNMT
```

And you are ready to go! Take a look at the [quickstart](quickstart.md) to familiarize yourself with the main training workflow.

## Docker (Ubuntu)

First you need to install `nvidia-docker`:

```bash
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb
```

If this command does not work, you may need to run the following updates:

```bash
sudo apt-add-repository 'deb https://apt.dockerproject.org/repo ubuntu-xenial main'
sudo apt-get update
sudo apt-get install docker-engine nvidia-modprobe
```

Then simply run our Docker container:

```bash
sudo nvidia-docker run -it harvardnlp/opennmt:8.0
```

Once in the instance, check out the latest code:

```bash
git clone https://github.com/OpenNMT/OpenNMT
```

## Amazon EC2

The best way to do this is through Docker. We have a public AMI with the preliminary CUDA drivers installed: `ami-c12f86a1`. Start a P2/G2 GPU instance with this AMI and run the `nvidia-docker` command above to get started.
