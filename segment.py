#author: Ben Hixon
#date: 9-21-2011
#Course: Advanced Algorithms in 3D Computer Vision
#Project 0: Range Image Segmentation
#description: Given a 3d point cloud, assign planar surfaces different colors.
#
#   We do this in the following way:
#       Classify each point by its local plane. A point is planar iff its kxk neighborhood has a good-fitting plane.
#       Two points are locally coplanar iff they have the same (or close enough) normal. Local normal is defined by the normal of the 
#       best-fitting plane in a kxk neighborhood.
#
#   (1) First, we load an image file into an Image object. This object contains a 2-d array of Point objects. Each Point object just contains the coordinates,
#   color, type (planar, nonplanar, or unknown), and equivalence class label (indicating what plane it belongs to; all points start out with an eclass of -1, which
#   means unlabeled.)
#
#   (2) Once we have the image in a data structure we call are_locally_coplanar() on every point. This function decides whether the point is fit by a kxk neighborhood
#    by the following:
#               (A) find the centroid of these kxk points
#               (B) find the difference between each point and the centroid, multiply the result with its transpose, and summing up those products to get matrix A
#               (C) find smallest eigenvalue of resulting A
#               (D) The normal is the eigenvector of A corresponding to that eigenvalue
#               (E) if eigenvalue is below a certain user-defined threshold then these points are coplanar by this measure and return True

#   (3) If are_locally_coplanar returns false, we say it's nonplanar and move on to the next point. But if it returns true, we say that the point is planar and the points 
#   are neighbors for the purposes of the sequential labeling algorithm. This algorithm then assigns the point to an equivalence class (which represents a distinct plane) 
#   in the following way:
#               (A) If its NW neighbor is labeled, give it that label.
#               (B) Else if its N and W neighbors are both labeled, give it the N's label and add both to the same equivalence class.
#               (C) Else if only one of N or W is labeled, give it that label
#               (D) Else, give the point a new label and assign it to its own equivalence class.
#
#   (4) Then we just assign each equivalence class its own random color = array([ random.randint(0,255) , random.randint(0,255) , random.randint(0,255) ]), and print them
#   to a file. I'm outputting a .pts file instead of a .ptx because my understanding is that .pts only has four elements per row, while the .pts has six, the last three being
#   the color.  I also output a boundary.pts giving the nonplanar points in red and the planar points in blue.
#
#   References: Geometry and Texture Recovery of Scenes of Large Scale, Ioannis Stamos and Peter K. Allen, CVIU 88, 94-118, 2002.
#
#   Comments on output:
#       If the threshold is set very low, then all points will be unlabeled, which means they'll all have the same color.
#       If the threshold is very high, then we get very few distinct planes (eg at threshold = 0.1, the small example is all the same color except for a few points).
#
#       A threshold of pthresh = 0.00001 was too small, even for the small example: every point was a 'boundary' point.  
#       Thresholds of 0.0001 and 0.001 produce same output on the small example. 0.0001 finds too many boundary points on big example,
#       but 0.001 doesn't always distinguish between two different planes, so the best threshold for the big example if it exists must be in between 
#       those two values. I used 0.0005 which seemed to give good results the large image and worked fine on the small one too.
#
#       The small example looks like only three planes, but due to the discontinuities in the larger plane, the larger plane itself is broken up into
#       smaller planes. I could not get rid of this effect using only this algorithm, since the discontinuities are 'stronger' than the border between the
#       larger plane and the adjacent, second-largest plane: at a larger threshold, that legitimate border disappears before the discontinuities on the largest plane do.
#
#

import string, sys
from numpy import *
from numpy.linalg import *

class Image:
    '''This class is the "2.5d" image array. It's a 2d array of point objects. Each point object is also a class, implemented below, 
    but is basically just a pair of two lists, one for the x,y,z coords and one for the color label. I also gave each point a 'neighborhood' label,
    to be used by the sequential labeling algorithm. The Image class includes useful methods like returning the kxk neighborhood about a point.
    '''
    
    def __init__(self,filename,skiplines):
        
        print("opening file")
        lines = open(filename, "r").readlines()
        print("done opening file and putting into lines list")
        
        self.rows = rows = int(lines[0])
        self.cols = cols = int(lines[1])
        self.image = []
        color = array([-1,-1,-1])   #'color'; the initial [-1,-1,-1] means that it is unlabeled.
        
        n=skiplines     #start reading from lines here
        for i in range(0,rows):             # actually the pts file is column1,column2,column3,etc, so this needs to be flipped. though it makes no difference
            self.image.append([])       #create empty list on row i
            for j in range(0,cols):     #this is big-O(n^2), ie proportional to size of the image array.
                l = lines[n].split(' ')
                coords = array([float(l[0]),float(l[1]),float(l[2])] )
                self.image[i].append( Point(coords,color) )
                n=n+1
                
        self.image = array(self.image)      #turn image into a ndarray -- is this okay?!
    
    def get_point(self,row,col):
        #returns a pair (array,list): the array is the x,y,z; the list is the color r,g,b.  Returns a copy not a reference. not used much.
        return self.image[row][col]     
    
    def get_kxk_neighborhood(self,row,col,k):
        '''Returns the k by k neighborhood centering around the point at (row,col). k should be odd.  The return type is an array list of numpy arrays.
                        
            right now I'm assuming k=3. fix this.
        '''
    
        return array([      [ self.image[row-1][col-1].coords,   self.image[row-1][col].coords,   self.image[row-1][col+1].coords  ],
                            [ self.image[row][col-1].coords,     self.image[row][col].coords,     self.image[row][col+1].coords    ],
                            [ self.image[row+1][col-1].coords,   self.image[row+1][col].coords,   self.image[row+1][col+1].coords  ]  ])              

class Point:
    '''the Image object holds a 2d array of these. 
    Contains x,y,z coords, r,g,b color, and other variables like 'type' (planar, nonplanar, unknown)'''
    
    def __init__(self,coords=array([0,0,0]),color=array([-1,-1,-1])):        
    
        self.coords = coords      
        self.color = color     
        
        self.eclass_label = -1      #equivalency class for use in sequential labeling algorithm...-1 means 'unlabeled'
        
        self.type='unknown'              #planar, nonplanar, or unknown. unknown means it's not yet classified; nonplanar means it's on a discontinuity.
        self.is_boundary = False          #boundary point is one for which at least one of its 8-neighbors is not coplanar with it (not in a 'cluster')
        
        #self.normal = array([0.,0.,0.])     #this will be calculated during planar classification. only for planar points. We don't need it here.
        
       
        
class PointCluster:     #NOT USED
    '''The Stamos-Allen algorithm first classifies points into local planes, and then groups these into point clusters using hough transform, and only then connects the clusters using
    the sequential labeling algorithm.  Probably won't do that in this assignment but if I do, here's an object for a point cluster.'''
    
    def ___init___(self, points):
        '''(not necessarily square?)'''
    
        self.points = points        #points is a list of Point objects
        size = len(points)
        

               
class UnionFind:
    '''My naive equivalence class was woefully inefficient (see above). Here's a better approach, an implementation of the Union-Find algorithm.
        The algorithm is described in Mark Allen Weiss' "Data Structures and Algorithm Analysis" and is almost constant time.  I also looked at 
        http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py and http://stackoverflow.com/questions/3067529/looking-for-a-set-union-find-algorithm 
        for suggestions for python implementations and followed the second one pretty closely. I'm surprised that Python doesn't have this object natively.
        '''
        
    def __init__(self):
        '''initalize'''
        self.leader = {}        #dictionary that given key=label, returns leader of label's eclass
        self.group = {}         #given a leader, return the partition set it leads
        
    def add(self,a,b):
        '''add a and b to the object. if they're already in the same eclass do nothing. If both belong to different eclasses, merge 
        the smaller eclass into the larger eclass and delete the smaller, and make sure every former member of the smaller knows its
        new leader. If only one of a and b is in an eclass, but the singleton into the other's eclass. If neither have an eclass, just
        make a new one with them as members and a as the leader.'''
        
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return       #these aren't distinct groups, quit
                groupa = self.group[leadera]        
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):       #make a the larger set (just flip around)
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb        #a's group now includes everything in b; '|' is set-union in python
                del self.group[leaderb]
                for k in groupb:        #reassign leader for every member of b since it's been merged into a
                    self.leader[k] = leadera
            else:
                #if a's group isn't empty but b's group is, just stick b into a's group
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                #if a's group is empty but b's group isn't, stick a into b's group
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                #both a's group and b's group were empty, so make a new group and stick them both into it with a as leader
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])
        
    def make_new(self,a):
        '''add a new singleton partition with a as the leader and sole member'''
        self.leader[a]=a
        self.group[a] = set([a])        #a now leads a new singleton set
                
def are_locally_coplanar(P, p_thresh):
    #P is a kxk array (neighborhood) of points. each point is a numpy ndarray.
    #this function returns true if they're locally coplanar, false if they're not.
    #We decide if they're coplanar by:
    #   (a) find the centroid of these kxk points
    #   (b) find the difference between each point and the centroid, multiply the result with its transpose, and summing up those products to get A
    #   (c) find smallest eigenvalue of resulting A
    #   (d) normal is the eigenvector of A corresponding to that eigenvalue
    #   (e) if eigenvalue is below a certain user-defined threshold then these points are coplanar by this measure
    
    k = len(P[0])
    
    centroid = array([0.,0.,0.])           
    for i in range(0,k):
        for j in range(0,k):        #9 calls when k=3
            centroid = centroid + P[i][j]
   
    centroid = centroid/(k*k)   #there's k*k points in the neighborhood; centroid is average of these.
    
    A = array([[0,0,0],[0,0,0],[0,0,0]])        #will store the sum of the differences between each point and the centroid multiplied by their transpose
    
    #for each point P[i][j] in this kxk neighborhood, subtract centroid from it, multiply that difference with its transpose, and add result to the matrix A.
    for i in range(0,k):
        for j in range(0,k):
            [p1,p2,p3] = P[i][j] - centroid                      #[p1,p2,p3] = point minus centroid (given as p' or p-prime in notes)
            A = A + array([[p1],[p2],[p3]])*array([p1,p2,p3])           # A = A + ([p1,p2,p3]^T times [p1,p2,p3])
                
    #local normal for this neighborhood is the eigenvector for the smallest eigenvalue of the matrix A, 
    #where A is the sum of the difference vectors between each point and the centroid.
    
    eigs = eigh(A)               #Eigenvalues and corresponding eigenvectors of A. See http://docs.scipy.org/doc/numpy-1.3.x/reference/generated/numpy.linalg.eig.html
                                 #We use eigh to only get real eigenvalues.
    eigenvalues = eigs[0]       #array of eigenvalues of A, unsorted
    eigenvectors = eigs[1]      #array of eigenvectors of A, normalized, ordered according to their corresponding eigenvalue in eigs[0]
    min_eval_index = argmin(eigenvalues)     #min_eval_index is the *index* of the smallest eigenvalue
            #this is going to be imaginary sometimes? but not if we use eigh(A) instead of eig(A)
    min_eigenval = eigenvalues[min_eval_index]  #smallest eigenvalue
    normal = eigenvectors[min_eval_index]     #The eigenvector associated with the smallest eigenvalue is the normal. (We don't actually need this...)
        
    if(min_eigenval <= p_thresh): return True   #if the eigenvalue is small enough, these points are coplanar so return True.
    else: return False
    

#   * * * * * BEGIN MAIN * * * * *

    
# VARIABLES     #make these command-line arguments
k = 3                #we fit the plane to the k x k neighborhood of points
p_thresh = 0.0001    #threshold of eigenvalue for which points are considered locally planar. if eigenvalue>=p_thresh, then points are not planar.
                     #All thresholds of 0.01 and higher produce the same output, down to the same number of equivalency classes.
                     #threshold of 0.0001 and 0.001 produce same output on the small example. 0.0001 finds too many boundary points on big example,
                     #but 0.001 doesn't always distinguish between two different planes, so the best threshold if it exists must be in between 
                     #those two values. I used 0.0005 which seemed to give good results the large image and worked fine on the small one too.
                     
                     #However, even 0.0005 was too large for the NECorner; 0.0001 worked well.
                   
#filename = 'small_example.pts'         
#filename = 'big_example.pts'
filename = 'NECorner.pts'  
skiplines = 10        #number of garbage rows at beginning (including rows and columns). unique to each file unfortunately.  should be 2 for small and big example, 10 for necorner

#READ INTO ARRAY
print("reading input into image array object")
I = Image(filename,skiplines)           #create Image object from this file.
print("done reading input into image array object")

label=0
E = UnionFind()     #equivalence class object to use with seq labeling algorithm

# SEQUENTIAL LABELING ALGORITHM                   
print("calling are_coplanar() on every point's kxk neighborhood")
for row in range(1,I.rows-1):
    for col in range(1,I.cols-1):       #There's two bottlenecks in this loop: calling are_locally_coplanar, and adding a label to an eclass.
        #for each point on the image, we get it's 3x3 neighborhood, and ask if these 9 points are coplanar. If so, label the middle point as follows:
        #   first, if NW neighbor is labeled, give it the NW neighbor's label
        #   Else, if N and W are both labeled but differently, give it the N's label and add N and W to same equivalence class
        #   Else, if only one of N or W are labeled, give it that label
        #   Finally, if neither N,W, or NW are labeled, create a new label, give it to the middle point, and add it to a new equivalence class
                        
        P = I.get_kxk_neighborhood(row,col,k)       #P is the 3x3 neighborhood centered about (row,col).
                        
        if( are_locally_coplanar(P, p_thresh) and I.get_point(row,col).coords.any() ):        #if all points in P are locally coplanar and middle point isn't zero, 
                                                                                              
            I.image[row][col].type='planar'      #it's locally coplanar with its k-neighborhood, so it's a planar point      
            
            #labels of N, W, and NW points. If these are zero or unlabeled, the value will be -1.
            p = I.get_point(row,col).eclass_label           #p (should be -1)
            N = I.get_point(row-1,col).eclass_label         #up
            W = I.get_point(row,col-1).eclass_label         #left
            NW = I.get_point(row-1,col-1).eclass_label      #top left
                                    
            #Sequential labeling algorithm:
            #   if NW is labeled, set to that one's label
            #   else if both N and W are labeled, set to N's label and add both N&W to an equivalence class
            #   else if either N or W is labeled, set to that label
            #   else make a new label and add to its own new eclass
            
            if(NW is not -1):   #if nw point is labeled, give the middle point the same label
                I.image[row][col].eclass_label = NW   
            elif( (N is not -1) and (W is not -1) ):    #both are labeled, so set to N label and add both to the same equiv class
                I.image[row][col].eclass_label = N      
                E.add(N,W)
            elif(N is not -1): I.image[row][col].eclass_label = N   #if only N or W is labeled, assign it that label
            elif(W is not -1): I.image[row][col].eclass_label = W 
            else:   #none were labeled so make new label for this point and give it a new eclass  (this point is on its own new corner)
                label+=1
                I.image[row][col].eclass_label = label
                E.make_new(label)
        else:   #if they're not coplanar, then middle point doesn't fit a local plane and so make it a nonplanar 'boundary' point
            I.image[row][col].is_boundary=True
            I.image[row][col].type='nonplanar'
        
        if(col%100 is 0 and row%40 is 0):print("we're on point (row=",row,",col=",col,")")
#end for loop through image.
print("we're done with classifying based on local planarity")
 
print("starting equivalency color labeling (all points with a label in the same eclass get the same random color)...") 
    
#assign a random color to each partition:    
class_colors = {}
for e in E.group.keys():
    random.seed()
    color = array([ random.randint(0,255) , random.randint(0,255) , random.randint(0,255) ])
    class_colors[e] = color         #class_colors is a dictionary that returns a color when passed a partition leader
    
#for each point in the image, assign it the color of its partition by getting its leader and passing the leader to class_colors    
for row in range(0,I.rows):         
        for col in range(0,I.cols):
            if(I.get_point(row,col).coords.any() and I.get_point(row,col).eclass_label is not -1):      #if it's a planar nonzero, give it a color
                eclass = E.leader[ I.get_point(row,col).eclass_label ]
                I.image[row][col].color = class_colors[ eclass ]
            
print("...done with equivalency relabeling!")

#   OUTPUT TO FILE
out_lines = []
out_lines.append(I.rows)
out_lines.append(I.cols)
line = ""
for row in range(1,I.rows-1):   #unroll image, convert its elements to strings, and append to out_lines which is saved to file
        for col in range(1,I.cols-1):            
            if( I.get_point(row,col).coords.any() ):        #only append non-zeroes (since they contribute nothing except file size)
                #turn image entry back into a string and append string to out_lines)
                line1 = " ".join([str(x) for x in I.get_point(row,col).coords])      #x,y,z
                line2 = " ".join([str(x) for x in I.get_point(row,col).color])      #color
                line = line1+' '+line2
                out_lines.append(line)            
ofname = 'out.pts'
with open(ofname, mode='w') as fo:
        for l in out_lines:
            fo.write(str(l)+'\n')
          
#draw boundaries and save to boundaries.pts:
out_lines=[]
out_lines.append(I.rows)
out_lines.append(I.cols)
line = ""
for row in range(0,I.rows):
        for col in range(0,I.cols):            
            if( I.get_point(row,col).coords.any() ):        #only append non-zeroes (since they contribute nothing except file size)
                #turn image entry back into a string and append string to out_lines)
                line1 = " ".join([str(x) for x in I.get_point(row,col).coords])      #x,y,z
                #line2 = " ".join([str(x) for x in I.get_point(row,col).color])      #color, not needed here since I want a binary image showing boundaries
                if(I.get_point(row,col).is_boundary): line2 = "255 0 0"
                else: line2 = "0 0 255"
                line = line1+' '+line2
                out_lines.append(line)            
ofname = 'boundaries.pts'
with open(ofname, mode='w') as fo:
        for l in out_lines:
            fo.write(str(l)+'\n')

# * * * * * END MAIN * * * * *
            