import numpy as np
import math
import linecache


class Batch_Generator(object):
    def __init__(self):
        self.voc2id = {}
        self.id2voc = {}
        self.id2freq = {}
        self.de2id = {}
        ##self.Entry = []
        self.words = np.zeros(1000)
        ##self.deps = [[[]]]
        self.num_wrds = 0
        self.num_deps = 0
        self.num_wsd = 0
        self.src = 0
        self.dest = 0
        self.lbl = 0
        self.cnt_edges = 0
        self.cnt_wrds = 0
        self.cnt_negs = 0
        self.cnt_sample = 0
        self.target = 0
        self.voc_size = 0
        self.b_elen = 0
        self.b_wlen = 0
        self.table_size = int(1e8)
        self.train_words = 0
        self.next_random = 0
        self.read_ids()

    def InitUnigramTable(self, voc_size):
        a, i = 0, 0
        train_words_pow = 0.
        d1 = 0.
        power = 0.75;
        self.table = np.zeros(self.table_size)
        for a in range(voc_size):
            train_words_pow += pow(self.id2freq[a], power);
        i = 0;
        d1 = pow(self.id2freq[i], power) / train_words_pow;
        for a in range(self.table_size):
            self.table[a] = i;
            if (a / self.table_size > d1):
                i += 1
                d1 += pow(self.id2freq[i], power) / train_words_pow
            if (i >= voc_size): i = voc_size - 1

    def read_ids(self):
        # Reading voc2id
        with open("./data/voc2id.txt",
                  "r") as fin:
            while fin:
                line = fin.readline()
                if len(line.strip()) == 0:
                    break
                tokens = line.split('\t')
                self.voc2id[tokens[0]] = int(tokens[1])
                self.id2voc[int(tokens[1])] = tokens[0]
        voc_size = len(self.voc2id)

        # Reading id2freq
        with open("./data/id2freq.txt",
                  "r") as fin:
            while fin:
                line = fin.readline()
                if len(line.strip()) == 0:
                    break
                tokens = line.split('\t')
                self.id2freq[int(tokens[0])] = int(tokens[1])
                self.train_words += int(tokens[1])

        # Reading de2id
        with open("./data/de2id.txt",
                  "r") as fin:
            while fin:
                line = fin.readline()
                if len(line.strip()) == 0:
                    break
                tokens = line.split('\t')
                self.de2id[tokens[0]] = int(tokens[1])
        self.InitUnigramTable(len(self.voc2id))

    def reset(self):
        self.freq = 0;
        self.fin = open("./data/data.txt",
                        "r")

    max_len, cntxt_edge_label = 0, 0

    def getBatch(self,
                 edges,  # // Edges in the sentence graph
                 wrds,  # // Nodes in the sentence graph
                 neg,  # // Negative samples
                 sub_samp,  # // Subsampling
                 elen,  # // Edges length
                 wlen,  # // Word length
                 win_size,  # // Window size for linear context
                 num_neg,  # // Number of negtive samples
                 batch_size,  # // Batchsize
                 sample,  # // Paramter for deciding rate of subsampling
                 random_line
                 ):

        cnt_edges, cnt_wrds, cnt_negs, cnt_sample = 0, 0, 0, 0  # // Count of number of edges, words, negs, samples in the entire batch
        cntxt_edge_label = len(self.de2id)
        for i in range(batch_size):
            #print('batch_generator.py i={}'.format(i))
            b_elen, b_wlen = 0, 0  # // Count of number of edges and word in particular element of batch
            line = linecache.getline("./data/data.txt", random_line).split()  ### Note: first line is 1, not 0
            random_line += 1
            # if random_line == 56790634:
            #      return 1
            # line = self.fin.readline().split()
            j = 0
            self.num_wrds, self.num_deps, self.num_wsd = int(line[0]), int(line[1]), int(line[2])
            ##not sure what self.num_wsd is
            j += 2
            while j - 2 < self.num_wrds:
                # print('j in getBatch=' , j)
                # if j == 16:
                # 	print('debug')
                wid = int(line[j])
                j += 1
                wrds[cnt_wrds] = wid
                cnt_wrds += 1
                b_wlen += 1
                if (sample > 0):  # // Performing subsampling
                    ran = (math.sqrt(self.id2freq[wid] / (sample * self.train_words)) + 1) * (
                                sample * self.train_words) / self.id2freq[wid];
                    self.next_random = self.next_random * 25214903917 + 11;
                    if (ran < (self.next_random & 0xFFFF) / 65536):
                        sub_samp[cnt_sample] = 0;
                    else:
                        sub_samp[cnt_sample] = 1;
                    cnt_sample += 1;

                k = 0;
                while (k < num_neg):  # // Getting negative samples
                    self.next_random = self.next_random * 25214903917 + 11;
                    target = int(self.table[(self.next_random >> 16) % self.table_size]);
                    if (target == wid): continue;
                    neg[cnt_negs] = target;
                    cnt_negs += 1
                    k += 1

            while j - 2 - self.num_wrds < self.num_deps:  # // Including dependency edges
                sdl = line[j]
                j += 1
                sdl_arr = sdl.split('|')
                src, dest, lbl = int(sdl_arr[0]), int(sdl_arr[1]), int(sdl_arr[2])
                edges[cnt_edges * 3 + 0] = src;
                edges[cnt_edges * 3 + 1] = dest;
                edges[cnt_edges * 3 + 2] = lbl;
                cnt_edges += 1
                b_elen += 1
            wlen[i] = b_wlen
            elen[i] = b_elen
        return 0
