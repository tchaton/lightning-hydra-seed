python -c "print(''.join([l.split(';')[0]+'\n' if ('python_version' in l or 'sys_platform' in l) else l for l in open('requirements.txt', 'r').readlines()]))" > requirements_tmp.txt
mv requirements_tmp.txt requirements.txt
