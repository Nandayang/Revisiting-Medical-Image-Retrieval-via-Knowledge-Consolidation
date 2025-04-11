import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict


def inference_ACIR(dataloader, net, device, act):
    fids, hash, cls_label, cls_pd = [], [], [], []
    net.eval()
    for img, _, cls, _, _ in dataloader:
        cls_label.append(cls)
        hashcode, pd_cls_logits = net(img.to(device))
        if act=='tanh':
            hash.append(torch.tanh(hashcode).sign().data.cpu())
        elif act==None:
            hash.append(hashcode.sign().data.cpu())
        cls_pd.append(pd_cls_logits)
    return torch.cat(hash).cpu().numpy(), torch.cat(cls_label).cpu().numpy(), torch.cat(cls_pd).cpu().numpy()


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def compute_topK(trn_hash, tst_hash, trn_label, tst_label, radius, number_class, topK=[100, 500, 1000]):
    AP = defaultdict(list)
    AR = defaultdict(list)
    smooth = 1e-7
    trn_label = np.argmax(trn_label, axis=-1)
    tst_label = np.argmax(tst_label, axis=-1)
    kmap, kmar = [],[]
    for k in topK:
        for r in radius:
            for i in range(number_class):
                AP['AP{}'.format(r)].append([])
                AR['AR{}'.format(r)].append([])
        for i in range(tst_hash.shape[0]):
            query_label, query_hash = tst_label[i], tst_hash[i]
            distance = np.sum((query_hash != trn_hash), axis=1)

            argidx = np.argsort(distance)[:k]
            # precision=TP/(TP+FP) recall=TP/(TP+FN)
            buffer_yes = (query_label == trn_label[argidx])
            buffer_1_0 = np.stack([buffer_yes, 1 - buffer_yes])

            for r in radius:
                if r == 0:
                    TPFP = ((distance[argidx] == 0) * buffer_1_0).sum(axis=1)
                    FN = ((distance[argidx] != 0) * buffer_yes).sum()
                else:
                    TPFP = ((distance[argidx] <= r) * buffer_1_0).sum(axis=1)
                    FN = ((distance[argidx] > r) * buffer_yes).sum()

                # print("TP: {}, FP:{}, FN:{}".format(TPFP[0], TPFP[1], FN))
                AP['AP{}'.format(r)][query_label].append(TPFP[0] / (TPFP.sum() + smooth))
                AR['AR{}'.format(r)][query_label].append(TPFP[0] / (TPFP[0] + FN + smooth))
        mapL = []
        marL = []
        for r in radius:
            ap, ar = [], []
            for j in range(number_class):
                ap.append(np.array(AP['AP{}'.format(r)][j]).mean())
                ar.append(np.array(AR['AR{}'.format(r)][j]).mean())
            # print("AP{}: {}|{}".format(r, np.round(np.array(ap).mean(),4), np.round(np.array(ap).std(),4)))
            # print("AR{}: {}|{}".format(r, np.round(np.array(ar).mean(),4), np.round(np.array(ar).std(),4)))
            mapL.append(np.round(np.array(ap).mean(), 4))
            marL.append(np.round(np.array(ar).mean(), 4))
            # print("=====================================")
        max_map = np.max(mapL)
        max_mar = marL[np.argmax(mapL)]
        print("top {}, map {}".format(k, max_map))
        kmap.append(max_map)
        kmar.append(max_mar)
    return np.max(kmap), kmar[np.argmax(kmap)], np.argmax(mapL)


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    """
    Calculate Top-K Mean Average Precision (mAP).

    Args:
        rB (numpy.array): Retrieval binary codes.
        qB (numpy.array): Query binary codes.
        retrievalL (numpy.array): Retrieval labels.
        queryL (numpy.array): Query labels.
        topk (int): Number of top results to consider.

    Returns:
        float: The computed Top-K mAP.
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for i in tqdm(range(num_query)):
        gnd = (np.dot(queryL[i, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[i, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap += np.mean(count / tindex)
    topkmap /= num_query
    return topkmap