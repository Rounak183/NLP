

import csv
import itertools
import sys

fields = [ 'org', '2015', '2014', '2013' ]
dw     = { 'orgname1': { '2014' : 2, '2015' : 1, '2013' : 1 },
           'orgname2': { '2015' : 1, '2014' : 2, '2013' : 3 },
           'orgname3': { '2015' : 1, '2014' : 3, '2013' : 1 }
        }

fold=2
string="Output"+str(fold)+".csv"
w = csv.DictWriter( sys.stdout, fields )
for key,val in sorted(dw.items()):
    row = {'org': key}
    row.update(val)
    w.writerow(row)