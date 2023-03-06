import re

from configs import exps_idx, faults_idx

def filter_key(keys):
    fkeys = []
    for key in keys:
        matchObj = re.match( r'(.*)FE_time', key, re.M|re.I)
        if matchObj:
            fkeys.append(matchObj.group(1))

    return fkeys[0]+'DE_time',fkeys[0]+'FE_time'

def get_class(exp, fault):
    if fault == 'Normal':
        return 0
    return exps_idx[exp] + faults_idx[fault]