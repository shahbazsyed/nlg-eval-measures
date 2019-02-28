import os
import sys
sys.path.append(__file__)
import subprocess, threading
from subprocess import check_output
import tempfile

meteor_jar = os.path.abspath(os.path.join(os.path.dirname(__file__),'meteor-1.5.jar'))

def _create_files(peer, model):
    peer_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
    model_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
    with open(peer_file.name,'w') as pf, open(model_file.name, 'w') as mf:
        pf.write(peer)
        mf.write(model)
    return (peer_file.name, model_file.name)

def compute_meteor_score(peer, model):
    print(meteor_jar)
    tmp_files = _create_files(peer, model)
    command = ['java','-Xmx2G','-jar',meteor_jar,tmp_files[0],tmp_files[1],'-l','en','-norm']
    try:
        output = check_output(command, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
    return output

def compute_meteor_from_files(peer_file, model_file):
    command = ['java','-Xmx2G','-jar',meteor_jar,peer_file,model_file,'-l','en','-norm','-vOut']
    try:
        output = check_output(command, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
    return output

if __name__ == "__main__":
    peer = "I invaded my gfs privacy and discovered a fetish that I find mildly disturbing what the hell should I do?"
    model = "I looked into my gfs private stuff and discovered that she has a disturbing fetish."
    meteor_score = compute_meteor_score(peer, model)
    print(meteor_score)
    