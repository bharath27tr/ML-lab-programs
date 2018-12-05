import csv
hypo={'%','%','%','%','%','%'}
with open('input.csv') as csv_file:
	readcsv = csv.reader(csv_file,delimiter=',')
	data=[]
	print("the given training examples are:")
	for row in readcsv:
		print(row)
		if row[len(row)-1].upper()=="YES":
			data.append(row)

print("\n the positive examples are:")
for x in data:
	print(x)
print("\n");
TotalExamples = len(data);
i=0;

print("the steps of the Find-s algorithm are",hypo);
list = [];
p=0;
d=len(data[p])-1;
for j in range (d):
	list.append(data[i][j]);
hypo = list;

for i in range(TotalExamples):
	for k in range(d):
		if hypo[k]!=data[i][k]:
			hypo[k]='?';
		else:
			hypo[k];
	print(hypo);

print("\n the maximally specific Find-s hypothesis for the given training examples is");
list=[];
for i in range(d):
	list.append(hypo[i]);
print(list);
