#!/usr/bin/env python
# coding: utf-8

import argparse
import gzip, os, csv
import numpy as np
import random
import time
import networkx as nx
import gensim
import shutil

import findspark
findspark.init("/usr/local/spark")

from pyspark import SparkConf, SparkContext



def addTriple(net, source, target, edge):
    if source in net:
        if  target in net[source]:
            net[source][target].add(edge)
        else:
            net[source][target]= set([edge])
    else:
        net[source]={}
        net[source][target] =set([edge])
            
def getLinks(net, source):
    if source not in net:
        return {}
    return net[source]

def randomWalkUniform(triples, startNode, max_depth=5):
    next_node =startNode
    path = 'n'+str(startNode)+'->'
    for i in range(max_depth):
        neighs = getLinks(triples,next_node)
        #print (neighs)
        if len(neighs) == 0: break
        weights = []
        queue = []
        for neigh in neighs:
            for edge in neighs[neigh]:
                queue.append((edge,neigh))
                weights.append(1.0)
        edge, next_node = random.choice(queue)
        path = path+ 'e'+str(edge)+'->'
        path = path+ 'n'+str(next_node)+'->'

    return path



def preprocess(folders, filename):
    entity2id = {}
    relation2id = {}
    triples = {}

    ent_counter = 0
    rel_counter = 0
    for dirname in folders:
        for fname in os.listdir(dirname):
            if not filename in fname: continue
            print (fname)
            if fname.endswith('.gz'):
                gzfile= gzip.open(os.path.join(dirname, fname), mode='rt')
            else:
                gzfile =open(os.path.join(dirname, fname))

            for line in csv.reader(gzfile, delimiter=' ', quotechar='"'):
                #print (line)
                h = line[0]
                r = line[1]
                t = line[2]

                #if not t.startswith('<'): continue

                if h in entity2id:
                    hid = entity2id[h]
                else:
                    entity2id[h] = ent_counter
                    ent_counter+=1
                    hid = entity2id[h]

                if t in entity2id:
                    tid = entity2id[t]
                else:
                    entity2id[t] = ent_counter
                    ent_counter+=1
                    tid = entity2id[t]

                if r in relation2id:
                    rid = relation2id[r]
                else:
                    relation2id[r] = rel_counter
                    rel_counter+=1
                    rid = relation2id[r]
                addTriple(triples, hid, tid, rid)
            print ('Relation:',rel_counter, ' Entity:',ent_counter)
    return entity2id,relation2id,triples


def preprocess(file_path):
    entity2id = {}
    relation2id = {}
    triples = {}

    ent_counter = 0
    rel_counter = 0
   
    print (file_path)
    if file_path.endswith('.gz'):
        file= gzip.open(file_path, mode='rt')
    else:
        file =open(file_path)

    for line in csv.reader(file, delimiter='\t', quotechar='"'):
        #print (line)
        h = line[0]
        r = line[1]
        t = line[2]

        #if not t.startswith('<'): continue

        if h in entity2id:
            hid = entity2id[h]
        else:
            entity2id[h] = ent_counter
            ent_counter+=1
            hid = entity2id[h]

        if t in entity2id:
            tid = entity2id[t]
        else:
            entity2id[t] = ent_counter
            ent_counter+=1
            tid = entity2id[t]

        if r in relation2id:
            rid = relation2id[r]
        else:
            relation2id[r] = rel_counter
            rel_counter+=1
            rid = relation2id[r]
        addTriple(triples, hid, tid, rid)
    print ('Relation:',rel_counter, ' Entity:',ent_counter)
    return entity2id,relation2id,triples


def randomNWalkUniform(triples, n, walks, path_depth):
    path=[]
    for k in range(walks):
        walk = randomWalkUniform(triples, n, path_depth)
        path.append(walk)
    path = list(set(path))
    return path
    
def saveData(entity2id, relation2id, triples, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)  
    
    entity2id_file= open(os.path.join(dirname, 'entity2id.txt'),'w')
    relation2id_file = open(os.path.join(dirname, 'relation2id.txt'),'w')
    train_file = open(os.path.join(dirname, 'train2id.txt'),'w')

    train_file.write(str(num_triples)+'\n') 
    for source in triples:
        for  target in triples[source]:  
            hid=source
            tid =target
            for rid  in triples[source][target]:
                train_file.write("%d %d %d\n"%(hid,tid,rid))

    entity2id_file.write(str(len(entity2id))+'\n')  
    for e in sorted(entity2id, key=entity2id.__getitem__):
        entity2id_file.write(e+'\t'+str(entity2id[e])+'\n')  

    relation2id_file.write(str(len(relation2id))+'\n')    
    for r in sorted(relation2id, key=relation2id.__getitem__):
        relation2id_file.write(r+'\t'+str(relation2id[r])+'\n') 
        
    train_file.close()
    entity2id_file.close()
    relation2id_file.close()


def extractFeatureVector(model, drugs, id2entity, output): 
  
    header="Entity"
    ns = "n"

    for i in range(model.wv.vectors.shape[1]):
        header=header+"\tfeature"+str(i)
        
    fw=open(output,'w')
    fw.write(header+"\n")

    for id_ in sorted(drugs):
        nid =ns+str(id_)
        if  (nid) not in  model.wv:
            print (nid)
            continue
        vec = model.wv[nid]
        vec = "\t".join(map(str,vec))
        fw.write( id2entity[id_]+'\t'+str(vec)+'\n')
    fw.close()


def trainModel(drugs, id2entity, datafilename, model_output, vector_file, pattern, maxDepth):
    
    if not os.path.isdir(model_output):
        os.mkdir(model_output)

    
    output = os.path.join(model_output, pattern)
    if not os.path.isdir(output):
        os.mkdir(output)
    
    sentences = MySentences(datafilename, filename=pattern) # a memory-friendly iterator
    model1 = gensim.models.Word2Vec(size=200, workers=8, window=5, sg=1, negative=15, iter=5)
    model1.build_vocab(sentences)

    model1.train(sentences, total_examples=model1.corpus_count, epochs =5)
    modelname = 'Entity2Vec_sg_200_5_5_15_2_500'+'_d'+str(maxDepth)
    model1.save(os.path.join(output,modelname))
    
    extractFeatureVector(model1, drugs, id2entity, vector_file)
    del model1


class MySentences(object):
    def __init__(self, dirname, filename):
        self.dirname = dirname
        self.filename = filename

    def __iter__(self):
        print ('Processing ',self.filename)
        for subfname in os.listdir(self.dirname):
            if not self.filename in subfname: continue
            fpath = os.path.join(self.dirname, subfname)
            for fname in os.listdir(fpath):
                if not 'part' in fname: continue
                if '.crc' in fname: continue
                try:
                    for line in open(os.path.join(fpath, fname), mode='r'):
                        line = line.rstrip('\n')
                        words = line.split("->")
                        yield words
                except Exception:
                    print("Failed reading file:")
                    print(fname)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-tr',required=True, dest='rdf_file', help='enter folder path for rdf file')
    parser.add_argument('-w',required=True, dest='walk_folder', help='enter folder path for walks (seqence) to be stored')
    parser.add_argument('-kg',required=True, dest='kg_folder', help='enter folder path for knowledge graphs (as encoded n{#} for entities and e{#} for relation n) to be saved in relation2id and entity2id ')
    parser.add_argument('-m',required=True, dest='model_folder', help='enter folder path for model file to be stored')
    parser.add_argument('-v',required=True, dest='vector_file', help='enter path for model file to be saved')

    args = parser.parse_args()

    rdf_file= args.rdf_file
    walk_folder = args.walk_folder
    graph_folder = args.kg_folder
    model_folder = args.model_folder
    vector_file = args.vector_file



    findspark.init()
    if False: 
        sc.stop()

    config = SparkConf()
    config.setMaster("local[10]")
    config.set("spark.executor.memory", "70g")
    config.set('spark.driver.memory', '90g')
    config.set("spark.memory.offHeap.enabled",True)
    config.set("spark.memory.offHeap.size","50g") 
    sc = SparkContext(conf=config)
    print (sc)


    #fileext = '.nq.gz'
    entity2id, relation2id, triples = preprocess(rdf_file)

    num_triples=0
    for source in triples:
        for  target in triples[source]:
            num_triples+=len(triples[source][target])
    print ('Number of triples',num_triples)



    entities = list(entity2id.values())
    b_triples = sc.broadcast(triples)


    #folder = './walks/'
    folder = walk_folder
    if os.path.isdir(folder):
        shutil.rmtree(folder) 
    os.mkdir(folder)
    
    walks = 250
    maxDepth = 5
    for path_depth in range(1,maxDepth):
        filename = os.path.join(folder,'randwalks_n%d_depth%d_pagerank_uniform.txt'%(walks, path_depth))
        print (filename)
        start_time =time.time()
        rdd = sc.parallelize(entities).flatMap(lambda n: randomNWalkUniform(b_triples.value, n, walks, path_depth))
        rdd.saveAsTextFile(filename)
        elapsed_time = time.time() - start_time
        print ('Time elapsed to generate features:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    #dirname = './graph'
    os.mkdir(graph_folder)
    saveData(entity2id, relation2id, triples, graph_folder)

    print (len(entities))


    id2entity = { value:key for key,value in entity2id.items()} 

    #datafilename = './walks/'
    #model_output = './models/'  
    os.mkdir(model_folder)

    pattern = 'uniform'
    #vector_output =  './vectors/'
    trainModel(entities, id2entity, walk_folder, model_folder, vector_file, pattern, maxDepth)
