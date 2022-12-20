import tensorstore as ts
import numpy as np
import time

st = time.time()
edges = ts.open({
     'driver': 'zarr',
     'kvstore': {
         'driver': 'file',
         'path':'train_edges.bin.zarr'
     }
 },
         create=False,
         dtype=ts.int32,
         shape=[272115,3]).result()

embeddings = ts.open({
     'driver': 'zarr',
     'kvstore': {
         'driver': 'file',
         'path':'emb.zarr'
     }
 },
         create = True,
         delete_existing = True,
         dtype=ts.float64,
         shape=[14541,100]).result()

bs = 1000
loop_st = time.time()
np.random.seed(42)
for i in range(0,272000,bs):
    batch = edges[i:i+bs].read().result()
    unique_batch = np.unique(batch[:,[0,2]])
    batch_emb = embeddings[unique_batch]
    batch_emb += np.random.rand(*batch_emb.shape)
    embeddings[unique_batch] = batch_emb
    #wf = embeddings[unique_batch].write(batch_emb)
#wf.result()
et = time.time()
print(embeddings[:10,:2].read().result())
print('et - st', et - st)
print('et - loop_st', et - loop_st)
