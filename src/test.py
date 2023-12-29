from inpaint_test import start as inpaint_start
from segment_test import start as segment_start
import os
import time
import uuid

NAMESPACE_TEST = uuid.uuid3(uuid.NAMESPACE_DNS, "test")

for root, dirs, files in os.walk("dataset", topdown = False):
    for fn in files:
        uid = uuid.uuid3(NAMESPACE_TEST, fn)
        if fn.endswith(".png"):
            print("Deal with %s" % fn)
            filename = os.path.join(root, fn)
            #segment_start(filename, uid, int(time.time()))
            inpaint_start(filename, "segment/classes/%s_no" % uid, "segment/classes/%s_mask" % uid, uid, int(time.time()))
