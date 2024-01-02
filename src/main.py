# from pixelize_test import start as inpaint_start
from inpaint_test import start as inpaint_start
from segment_test import start as segment_start
from img2img_test import start as img2img_start
import os
import time
import uuid
import sys

NAMESPACE_TEST = uuid.uuid3(uuid.NAMESPACE_DNS, "main")

for root, dirs, files in os.walk(sys.argv[1], topdown = False):
    for fn in files:
        uid = uuid.uuid3(NAMESPACE_TEST, fn)
        if fn.endswith(".png"):
            print("Deal with %s" % fn)
            filename = os.path.join(root, fn)
            segment_start(filename, uid, int(time.time()))
            inpaint_start(filename, "segment/classes/%s_no" % uid, "segment/classes/%s_mask" % uid, uid, int(time.time()))
            img2img_start("inpaint-old/%s.png"%uid, str(uid), int(time.time()))
