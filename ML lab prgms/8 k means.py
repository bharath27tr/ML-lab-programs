import csv
import math
import copy
k=2

class cluster:
    def __init__(self,cluster_head_data):
        self.head=cluster_head_data
        self.l=[]
    def display_head(self):
        print(self.head)
    def add_ele_cluster(self,data):
        self.l.append(data)
    def display_ele(self):
        print('list contains',self.l)
        
def compare_the_values(first,secound):
    x=float(first[0])
    y=float(first[1])
    x1=float(secound[0])
    x2=float(secound[1])
    val=math.sqrt(  math.pow(math.fabs(x-x1),2)+math.pow(math.fabs(x-x1),2) )
    return val
        
def compare_the_nearest_cluster(cluster,data):
    #intialize the nearest cluster to 0
    dist_measure=None
    nearest=0
    for i in range(len(cluster)):
            dist=compare_the_values(cluster[i].head,data)
            if dist_measure is None:
                  dist_measure=dist
                  nearest=i
            if dist<dist_measure:
                  dist_measure=dist
                  nearest=i
    return nearest
                  



def recal_head(cluster):
    for i in range(len(cluster)):
        l1=cluster[i].l
        xval=0.0
        yval=0.0
        for j in l1:
            xval=xval+float(j[0])
            yval=yval+float(j[1])
        xavg=xval/len(l1)
        yavg=yval/len(l1)
        avgl=[]
        avgl.append(xavg)
        avgl.append(yavg)
        cluster[i].head=avgl

       
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
print('initial cluster head values')
#print display heads
for i in range(k):
    print('----cluster',i,'------')
    c[i].display_head()


        
error_ratio=1
# Iteration and including elelments in the cluster
while error_ratio>0:
    prevc=copy.deepcopy(c)
    for ele in db:
        r=compare_the_nearest_cluster(c,ele)
        c[r].add_ele_cluster(ele)

    # display all the elements
    for clust in c:
        clust.display_ele()
    #recalculate the avg value
        recal_head(c)
        
    for i in range(k):
        print('----cluster',i,'------')
        c[i].display_head()
    #remove the ele of cluter head for the next iter
    for i in range(k):
        c[i].l=[]
    #calculate the error
        error_ratio=0
    for i in range(k):
        if c[i].head != prevc[i].head:
            error_ratio=error_ratio+1
#final cluster ele
    


    
    
        
