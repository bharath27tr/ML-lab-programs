import csv
import math
import copy
k=2

class cluster:
    def __init__(self,cluster_head_data):
        self.head=cluster_head_data
        self.l=[]
        self.mean=cluster_head_data[0]
        self.variance=1.0
    def display_head(self):
        print("m=",self.mean)
        print("v=",self.variance)
    def add_ele_cluster(self,data):
        self.l.append(data)
        print(self.l)
    def display_ele(self):
        print('list contains',self.l)

    
def find_insert_individual_cluster(cluster,element,n):
    ele=float(element[0])
    prob=[]
    
    for i in range(len(cluster)):
        pxb= 1/math.sqrt(2*3.142*cluster[i].variance)* math.exp(-1* ( ele - float(cluster[i].mean) )**2 / (2*float(cluster[i].variance)))      
        print('pxb  exact==',pxb)
        prob.append(pxb)
    print('prob elem',prob)
    bi_den=0
    for i in range(len(prob)):
        bi_den=bi_den+prob[i]*1/n
        print('bi den', bi_den)
    #insert ele in to the cluster-- ele+pxb+bi
    for i in range(len(cluster)):
        clust_data=[]
        clust_data.append(ele)
        bi=(prob[i]*1/n)/bi_den
        clust_data.append(bi)
        #add the contents on to the cluster
        cluster[i].add_ele_cluster(clust_data)

def recalculate_cluster_mean_variance(cluster):
    l1=cluster.l
    print('list enteries',l1)
    #recalculating mean
    mean_num=0.0
    mean_den=0.0
    var_num=0.0
    var_den=0.0
    for i in range(len(l1)):
        mean_num=mean_num+l1[i][0]*l1[i][1]
        mean_den=mean_den+l1[i][1]
    mean=mean_num/mean_den
    cluster.mean=mean
    #recalculating varaiance
    for i in range(len(l1)):
        var_num=var_num+l1[i][1]*(l1[i][0]-mean)**2
        var_den=var_den+l1[i][1]
    variance=var_num/var_den
    cluster.variance=mean
        
def find_nearest(cluster,ele):
    ele=float(ele[0])
    prob=[]
    nearest_prob=None
    index=1
    for i in range(len(cluster)):
        pxb= 1/math.sqrt(2*3.142*cluster[i].variance)* math.exp(-1* ( ele - float(cluster[i].mean) )**2 / (2*float(cluster[i].variance)))      
        print('pxb for cluster i',i,'=',pxb)
        if nearest_prob is None:
            nearest_prob=pxb
            index=i
        else:
            if nearest_prob < pxb:
                nearest_prob=pxb
                index=i
    print('index',index,'nearest_prob=',nearest_prob)
    cluster[index].l.append(ele)

#read the contents of CSV file
with open('cluster.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    #inserting elelments in to the list
    db=[]
    for row in spamreader:
        db.append(row)
    #creating individual cluster heads
#displaying elelments of the list
print('Db entries')
print(db)

#intialize the cluster
c=[]

#intialize the cluster heads
for i in range(k):
    new_clust=cluster(db[i])
    c.append(new_clust)
print('initial cluster Mean Variance')
#print cluster mean and variance
for i in range(k):
    print('----cluster',i,'------')
    c[i].display_head()
    
error_ratio=1
# Iteration and including elelments in the cluster
while error_ratio>0:
    error_ratio=0    
    prevc=copy.deepcopy(c)
    #estimation
    for i in range(len(db)):
        find_insert_individual_cluster(c,db[i],len(db))
    #recalculate cluster mean and variance
    for i in range(len(c)):
        recalculate_cluster_mean_variance(c[i])
    #display recalculated mean and varaiance of cluster
    for i in range(k):
        print('----cluster',i,'------')
        c[i].display_head()
    #clear all the values of the cluster list for the next iteration
        for i in range(k):
            c[i].l=[]
    #calculate the error
        error_ratio=0
        for i in range(len(c)):
            if abs(c[i].variance - prevc[i].variance)>0.1:
                error_ratio=error_ratio+1
#display the cluster elements
#clear all the values of the cluster list for the next iteration
for i in range(k):
    c[i].l=[]
#claculate the nearest prob for each elelment and include it in the resultant cluster
for i in range(len(db)):
    find_nearest(c,db[i])

#display cluster elelments
for i in range(len(c)):
    print(c[i].l)

