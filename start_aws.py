import subprocess
from time import sleep
import json

instance_id = "i-0b13b15018aa0a2d7"

start_cmd = "aws ec2 start-instances --instance-ids {0}".format(instance_id)
check_cmd = "aws ec2 describe-instances"
paste_cmd = 'echo "{0}" | xclip -sel clipboard'
login_cmd = 'ssh -i ~/Downloads/aws.pem ubuntu@{0}'
clip_cmd  = 'echo "{0}" | xclip -sel clipboard'

try:
    subprocess.call(start_cmd, shell=True)
except:
    print "already running"
    
while True:
    done = False
    out = subprocess.check_output(check_cmd, shell=True)
    t = json.loads(out)
    for r in t['Reservations']:
        for i in r['Instances']:
            if i['InstanceId'] == instance_id:
                if i['State']['Name'] != "running":
                    print i['State']
                    continue
                else:
                    ip_addr = i['PublicIpAddress']
                    print ip_addr
                    subprocess.call(clip_cmd.format(login_cmd.format(ip_addr)), shell=True)
                    done = True
            else:
                continue
    if done:
        break
    # print t
    sleep(0.1)
