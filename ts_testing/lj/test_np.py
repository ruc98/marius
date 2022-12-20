import numpy as np
import time

st = time.time()
edges = np.memmap('train_edges.bin', dtype='int32', mode='r', shape=(62094395,2))

embeddings = np.memmap('emb.bin', dtype='float64', mode='w+', shape=(4847571,100))

bs = 1000
loop_st = time.time()
np.random.seed(42)
for i in range(0,10001,bs):
    print(i)
    batch = edges[i:i+bs]
    unique_batch = np.unique(batch)
    batch_emb = embeddings[unique_batch]
    batch_emb += np.random.rand(*batch_emb.shape)
    embeddings[unique_batch] = batch_emb
    embeddings.flush()
et = time.time()
print(embeddings[:10,:2])
print('et - st', et - st)
print('et - loop_st', et - loop_st)
