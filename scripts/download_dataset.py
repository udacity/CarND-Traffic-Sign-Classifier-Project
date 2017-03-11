import urllib.request
import zipfile
import os

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip'

tmp, message = urllib.request.urlretrieve(url)

with zipfile.ZipFile(tmp) as zip:
    zip.extractall('../data')

os.remove(tmp)
