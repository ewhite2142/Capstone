from urllib import request
import pandas as pd
import os
from skimage import io

home = os.getenv("HOME")
capstone_folder = home + "/dsi/Capstone/"

ids = []
for kind in ['Mountains', 'Mounts', 'Peak']:
    print("\n{}".format(kind))
    numphotos = 0
    excel_file = capstone_folder + kind + "_Img.xls"
    images_folder = capstone_folder + kind + "/"
    df = pd.read_excel(excel_file)
    numrows = df.shape[0] - 1

    for i in range(numrows):
        image_id =  df.iloc[i]['Img Id']
        if image_id in ids:
            print("id {} is already in ids!!!".format(image_id))
        ids.append(image_id)
        fn = images_folder + str(image_id) + ".jpg"
        url = df.iloc[i]['link']

        maxtries = 10
        numtries = 0
        notread = True
        while notread == True:
            numtries += 1
            if numtries > maxtries:
                print"{} attempts to read image_id {} failed...skipping".format(maxtries, image_id))
                break

            with open(fn, 'wb') as f:
                f.write(request.urlopen(url).read())

            #confirm image downloaded correctly by reading it
            try:
                img = io.imread(fn)
            except Exception as ex:
                continue

            if img.shape[0] < 20 or img.shape[1] < 20:
                continue

            notread = False


        numphotos += 1
        if numphotos % 100 == 0:
            print("photo# {}".format(numphotos))
