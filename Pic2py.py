# -*- coding: utf-8 -*-
# @Time    : 12/07/2020
# @Author  : Yibo Li
# @Email   : lslsls0001@gmail.com
# @File    : Pic2py.py
# @Software: Anaconda

import base64


def pic2py(picture_name):
    """
    change the file format to be py
    :param picture_name:
    :return:
    """
    open_pic = open("%s" % picture_name, 'rb')
    b64str = base64.b64encode(open_pic.read())
    open_pic.close()
    # b64str needs to add with .decode()
    write_data = 'img = "%s"' % b64str.decode()
    f = open('%s.py' % picture_name.replace('.', '_'), 'w+')
    f.write(write_data)
    f.close()


if __name__ == '__main__':
    pics = ["slice.png"]
    for i in pics:
        pic2py(i)
    print("ok")