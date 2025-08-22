cat /etc/hosts | awk 'END{print $1}'  > master_ip
