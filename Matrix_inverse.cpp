#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <Windows.h>
#include <nmmintrin.h>
#include <math.h>

#include "QuadTree.h"
#include "MedianSearch.h"
#include "TypeDefs.h"
#include "PerfMonitor.h"
#include "NeuralNet.h"
#include "ConvNet.h"

#define SQ(x) ((x)*(x))
#define M_PI 3.141592654
#define FOR(index,s,f) for(s32 index = (s); index<(f); index++)
#define ArrLen(arr)(sizeof(arr)/sizeof(arr[0]))
#define clamp(currVal,maxVal)(((currVal)>(maxVal))* (maxVal) + (currVal)*((currVal)<=(maxVal)))


enum directions {LEFT,RIGHT};
enum orientations {HORIZONTAL,VERTICAL};

unsigned long seed = 123;
unsigned long z =512945829;
unsigned long Arand()
{
	unsigned long o_seed = seed;
	seed ^= seed << 2;
	seed ^= seed >> 9;
	seed ^= seed << 17;

	return seed ^ o_seed ^ z;
}

RGB black = {0,0,0};
RGB white = {255,255,255};
RGB blue  = {0,0,255};
RGB red  = {255,0,0};
RGB green = {0,255,0};
RGB yellow  = {255,255,0};
RGB cyan = {0,255,255};

struct gridInfo
{
	s32 width, height,
	gridW, gridH;
};

struct gridData
{
	s32 h,w;
	s32 chunkSize;
	u8*data;
};
gridData IgridData(s32 w ,s32 h,s32 chunkSize)
{
	gridData ret ={0};
	ret.w = w;
	ret.h = h;
	ret.chunkSize = chunkSize;
	ret.data = (u8*)malloc(w*h*chunkSize);
	return ret;
}
s32 getCellData_s32 (s32 i,s32 j,gridData gd)
{
	return ((s32*)(gd.data))[i*gd.w+j];
}

void FgridData(gridData& d)
{
	free(d.data);
	d.data = 0;
	d.h= 0;
	d.w = 0;
}


struct line_direction
{
	V2 position;
	V2 direction; // length
};


inline f32 dot(V2& a,V2 &b)
{
	return a.x*b.x + a.y*b.y;
}
inline V2 operator / (V2 a,V2 b)
{
	V2 ret = {a.x /b.x,a.y/b.y};
	return ret;
}
inline V2 operator *(f32 a,V2 b)
{
	V2 ret = {a*b.x,a*b.y};
	return ret;
}
inline void operator *=(V2& a,f32 b)
{
	a.x *=b;
	a.y *=b;
}
inline V2 operator +(V2&a,V2&b)
{
	V2 ret = {a.x+b.x,a.y+b.y};
	return ret;
}
inline V2 operator -(V2&a,V2&b)
{
	V2 ret = {a.x-b.x,a.y-b.y};
	return ret;
}

inline void operator +=(V2&a,V2&b)
{
	a.x+=b.x;
	a.y+=b.y;
}

union V4 
{
	struct {f32 x;
			f32 y;
			f32 z;
			f32 w;};
	f32 d[4];
};
union m4f32 
{
	struct{f32 a0,a1,a2,a3,
		       b0,b1,b2,b3,
		       c0,c1,c2,c3,
		       d0,d1,d2,d3;};
	 V4 v[4];
	 f32 d[4][4];
	 f32 df[16];
};
struct Z
{
	f64 r;
	f64 i;
};
Z inline operator +(Z a,Z b)
{
	Z ret = {a.r +b.r,a.i+b.i};
	return ret;
}
Z inline operator *(Z a,Z b)
{
	Z ret = {a.r*b.r - a.i*b.i,a.r*b.i + a.i*b.r};
	return ret;
}

f32 len(V2 &a)
{
	return sqrt(dot(a,a));
}
f32 len(Z a)
{
	return sqrt(a.i*a.i + a.r*a.r);
}
void writeBwBlockAsbmp(s32 w,s32 h,u8* data,s8 *);
void writeBlockAsbmp(s32 w,s32 h,u8* data,char * fileName);
void WriteImageAsbmp(image & img,char * fileName);
f32* generateRandomPairsOfFloat(s32,s32,s32,s32,s32);
u8* fillImgBasedOnNearestNeigboursResample(u8*data,s32,s32,s32,s32,s32,s32,s32);
u8* slowFillImgBasedOnNearestNeigboursResample(u8*,s32,s32,s32,s32,s32,s32,s32);
u8* produceSphereHeightImg(s32 width,s32 height);
void fillRegressionPallete(V2*pts,s32 point_count,s32* data,s32 w, s32 h);
u8 * createRegressionTest(s32 w,s32 h,s32 point_count);
V2 bezierCurvePoint(V2 a, V2 b ,V2 c ,V2 d,f32 t);
V2 quadraticCurvePoint(V2 a,V2 b,V2 c,f32 t);
V2 cubicCurvePoint(V2 a,V2 b,V2 c,V2 d,f32 t);
V2 cubicCurvePoint2(V2 a ,V2 b ,V2 c,V2 d ,f32 t);
void CreatePallateWithALineOnIt(V2 start,V2 end,f32 w,f32 h,u8*&,RGB);
void DrawPointRGB(V2 a,image &img,RGB color);
void DrawMendelbrotSet(Z z,f32 r,s32 iter,s32 &w,s32 &h,u8*&data,RGB &color);
void DrawTriangle(V2 A,V2 B,V2 C,s32 w ,s32 h,u8*&data,RGB color);
void DrawSirpensky(V2 A,V2 B,V2 C ,s32 n,s32 &w,s32 &h,u8*&data,RGB &color);
void DrawSirpensky(V2 A,V2 B ,s32 n,s32 &w,s32 &h,u8*&data,RGB &color);
void DrawAABB(BB bb,s32 w ,s32 h,u8*&data,RGB color);
void FillAABB(BB bb,s32 w ,s32 h,u8*&data,RGB color);
void InitPallate(f32 w, f32 h,u8* &pallate,s32 value=255);
void drawHistogram(V2 *points,s32 s ,s32 f,f32 widthPerBlock,f32 minY,f32 maxY,s32 DIR);
line_direction * CreateLineDirection(float * lengths,directions *dir,s32 count,orientations initialOr,V2 initialPos,f32 metersPerPixel);
void CreatePerpPolygon(line_direction * ld,s32 count,u8* &toFill,s32 w,s32 h);
void WriteNumber(char * str,s32 strLen,u8* img,s32 w,s32 h,V2 startPos,f32 scaleFactor);

inline m4f32 operator * (m4f32 &m0,m4f32 &m1)
{
	m4f32 ret = {0};
	s32 m,i,j;
	for (m = 0; m < 4; ++m)	
	{
		for (i = 0; i < 4; ++i)
		{	
			f32 cellValue = 0;
			for (j = 0; j < 4; ++j)
			{
				cellValue += m0.d[m][j]*m1.d[j][i];
			}
			ret.d[m][i] =cellValue;
		}
	}
	return ret;
}
void Perspective( 
    const f32 &angleOfView, 
    const f32 &imageAspectRatio, 
    const f32 &n, const f32 &f, 
    f32 &b, f32 &t, f32 &l, f32 &r) 
{
    f32 scale = tan(angleOfView * 0.5) * n; 
    r = imageAspectRatio * scale, l = -r; 
    t = scale, b = -t; 
}

m4f32 Frustum(const f32 &bottom,
 			  const f32 &top,
 			  const f32 &left,
 			  const f32 &right,
 			  const f32 &nearD,
 			  const f32 &farD)
{ 
	f32  A = -(farD+nearD)/(farD-nearD);
	f32  B = (-2*farD*nearD)/(farD-nearD);
	f32 _2n = 2*nearD;
	m4f32 ret =  { _2n/(right-left), 0         			, (right+left)/(right-left), 0,
	           	    0         			, _2n/(top-bottom)  , (top+bottom)/(top-bottom), 0,
	          	  	0         			,0                      ,A                       , B,
	          	    0         			,0                      ,-1                      , 0} ;

	return ret;
}

m4f32 CreatePerspectiveMat(float viewAngle,f32 aspectRatio,f32 nearD,f32 farD)
{
	f32 bottom, top, left, right;
	Perspective(viewAngle,aspectRatio, nearD, farD, bottom, top, left, right);
	return Frustum(bottom, top, left, right, nearD, farD);
}
m4f32 CreateUnPerspectiveMat(m4f32 PerspectiveMat)
{
	f32 a = PerspectiveMat.d[0][0],
			b = PerspectiveMat.d[0][2],
			c = PerspectiveMat.d[1][1],
		    d = PerspectiveMat.d[1][2],
			e = PerspectiveMat.d[2][2],
			f =  PerspectiveMat.d[2][3];
	m4f32 ret = {
					 		1.f/a,0    ,0    ,b/a,
							 0    ,1.f/c,0    ,d/c,
							 0    ,0    ,0    ,-1,
							 0    ,0    ,1.f/f,e/f };
	return ret;
}



void testProjectionUnProjection()
{
	m4f32 ret = CreateUnPerspectiveMat(CreatePerspectiveMat(M_PI/2,1,0.1f,1000))*CreatePerspectiveMat(M_PI/2,1,0.1f,1000);
	int x= 0;
}

V2 findCell(gridInfo g, V2 p)
{
	return V((g.gridW*(s32)p.x)/g.width,(g.gridH*(s32)p.y)/g.height);
}

gridData generateRandomGrid(s32 min_val,s32 max_val,gridInfo d)
{
	gridData ret = IgridData(d.gridW+1,d.gridH+1,sizeof(s32));
	s32* data = (s32*)ret.data;
	for(s32 i = 0; i<ret.h;i++)
	{
		for(s32 j = 0; j<ret.w;j++)
		{
			s32 index = i*ret.w +j;
			data[index] = (rand()+min_val)%(max_val+1);
		}
	}
	return ret;
}

m4f32 generateRandomGrid16(s32 count)
{
	m4f32 gridD = {   rand()%count,rand()%count,rand()%count,rand()%count,
					  rand()%count,rand()%count,rand()%count,rand()%count,
					  rand()%count,rand()%count,rand()%count,rand()%count,
					  rand()%count,rand()%count,rand()%count,rand()%count};
	return gridD;
}


f32 lerp(f32 a, f32 b,f32 t)
{
	return (1-t)*a + t*b;
}
f32 cosineInterpolation(f32 s, f32 f,f32 t)
{
  return lerp(s,f,(1.f-cos(t))/2.f);
}
f32 bilerp(f32 a, f32 b,f32 c ,f32 d, f32 t,f32 s,b8 cosine = 0)
{
	if(cosine)
		return lerp(cosineInterpolation(a,b,s),cosineInterpolation(c,d,s),t);
	else 
		return lerp(lerp(a,b,t),lerp(c,d,t),s);
}

enum NodeType{LR,UD};
enum pathDirection {left,right,up,down};

struct KdNode
{		
		NodeType NType;
		V2 data;
		b8 isFull;
		union
		{
			struct 
			{
				KdNode *left;
				KdNode *right;
			};
			struct 
			{	
				KdNode *up;
				KdNode *down;
			};
		};	
};

KdNode * KDNode(V2 val,NodeType type)
{
	KdNode * ret = new KdNode;//(KdNode*) malloc(sizeof(KdNode));
	memset(ret,0,sizeof(KdNode));
	ret->data= val;
	ret->isFull = true;
	ret->NType = type;
	return ret;
}

void insertIntoKdTree(KdNode &Tree,V2 val,NodeType type = LR)
{
	if(!Tree.isFull)
	{//first element;
		Tree.data = val;
		Tree.NType = type;
		Tree.isFull = true;
		return;
	}	
	KdNode *it = &Tree;
	KdNode *toAddTo = 0;
	pathDirection dir = left;
	while(it)
	{
		if(it->NType == LR)
		{
			if(it->data.x>val.x)
			{
				toAddTo = it;
				it = it->left;
				dir = left;
			}
			else
			{
				toAddTo = it;
				it = it->right;
				dir = right;
			}
		}
		else if(it->NType == UD)
		{
			if(it->data.y>val.y)
			{
				toAddTo = it;
				it = it->up;
				dir = up;
			}
			else 
			{
				toAddTo = it;
				it = it->down;	
				dir = down;
			}
		}
	}
	if(toAddTo)
	{
		switch(dir)
		{
			case left:
			case up:
				toAddTo->left = KDNode(val,type);
				break;
			case right:
			case down:
				toAddTo->right = KDNode(val,type);
				break;
		}
	}
}

void printVec(V2 vec)
{
	printf("%.3f,%.3f\n",vec.x,vec.y);
}

V2 findClosestPointRec(KdNode tree,V2 pt)
{
	if(tree.isFull)
	{
		f32 minDist = len(pt-tree.data);
		V2 closestPoint = tree.data;
		f32 rightDist = 1e6,leftDist = 1e6;
		if(tree.left)
			leftDist = len(pt-tree.left->data);
		if(tree.right)
			rightDist = len(pt-tree.right->data);
		if(leftDist> minDist && rightDist>minDist)
			return closestPoint;
		V2 rightBestPoint,leftBestPoint;
		if(tree.right)
		{
			rightBestPoint = findClosestPointRec(*tree.right,pt);
			rightDist = len(pt-rightBestPoint);
		}
		if(tree.left)
		{
			leftBestPoint = findClosestPointRec(*tree.left,pt);
			leftDist= len(pt-leftBestPoint);
		}
		f32 finalminDist = min(leftDist,min(rightDist,minDist));
		if(finalminDist == minDist)
			return closestPoint;
		else
		{
			if(leftDist == finalminDist)
				return leftBestPoint;
			else 
				return rightBestPoint;
		}
	}
}

V2 findClosestPoint(KdNode tree,V2 pt)
{
	f32 MaxValue = 1e10;
	f32 minDist = MaxValue;
	V2 minPt = V(0,0);
	KdNode * T = &tree;
	while(T)
	{
		if(T->isFull)
		{
			f32 currentLen = len(pt-(*T).data);
			minDist = min(currentLen,minDist);
			f32 leftLen = MaxValue,rightLen = MaxValue;
			if(T->left)
				leftLen = len(pt - T->left->data);
			if(T->right)
				rightLen = len(pt - T->right->data);
			if(currentLen < leftLen && currentLen < rightLen && currentLen <= minDist)
				return T->data;
			if(leftLen> minDist && rightLen > minDist)
				return minPt;
			else 
			{
				f32 minLength =  min(leftLen,min(currentLen,rightLen));
				if(minLength == leftLen)
				{
					minDist = minLength;
					minPt = (*T).left->data;
					if(T->left)
						T = T->left;
					else
						T = 0;
				}
				else if(minLength == rightLen)
				{
					minDist = minLength;
					minPt = T->right->data;
					if(T->right)
						T = T->right;
					else 
						T = 0;
				}
				else 
					return minPt;
				minDist = minLength;
			}
		}
		else break;
	}
	return minPt;
}

void swap(V2 * arr,s32 i,s32 j)
{
	assert(arr);
	V2 temp = arr[i];
	arr[i] = arr[j]; 
	arr[j] = temp;
}

void BubbleSortBy(V2 *points ,s32 s,s32 f,s32 DIR)
{
	FOR(i,s,f)
	{
		FOR(j,i+1,f)
		{
			if(points[i].d[DIR] > points[j].d[DIR])
					swap(points,i,j);
		}
	}
}
int partitionBy(V2 * arr, int start,int end,float seed,s32 DIR)
{	
	int i = start,j=end;
	while(i<=j)
	{
		if(arr[i].d[DIR]>=seed && arr[j].d[DIR]<=seed)
		{
			swap(arr,i,j);
			j--;
			i++;
		}
		if(arr[j].d[DIR] >seed)
			j--;
		if(arr[i].d[DIR] <seed)
			i++;
	}
	return j;
}

inline s32 pickPivot(V2 * points,s32 s,s32 f)
{
	if(f==0)
		return 0;
	return clamp(rand()%(f+1) + s,f);
}
#define STACK_SIZE 100000
s32 stack[STACK_SIZE][2];
s32 currentTop = -1;
b8 empty()
{
	return currentTop == -1;
}
void push(s32 new_s,s32 new_f)
{
	stack[++currentTop][0] = new_s;
	stack[currentTop][1] = new_f;
}
void pop(s32&new_s,s32&new_f)
{
	assert(!empty());
	new_s = stack[currentTop][0];
	new_f = stack[currentTop--][1];
}
void QsortBy(V2 * points, s32 s,s32 f,s32 DIR)
{
	if(!points || f <= s)
		return;
	s32 maxTop = 0;
	push(s,f); // initiate the process;
	while(!empty())
	{
		assert(currentTop < STACK_SIZE);
		if(maxTop<currentTop)
			maxTop = currentTop;
		s32 new_s,new_f;
		pop(new_s,new_f);
		if(new_s >= new_f)
			continue;
		if(new_f == new_s + 1 &&
			points[new_f].d[DIR] < points[new_s].d[DIR])
		{	
			swap(points,new_s,new_f);
			continue;
		}
		s32 pivotIndex = pickPivot(points,new_s,new_f);

#ifdef _DEBUG
		assert(bw(pivotIndex,new_s,new_f));
#endif
		s32 partitionIndex = partitionBy(points,new_s,new_f,points[pivotIndex].d[DIR],DIR);
		if(new_s == partitionIndex - 1 && 
			points[partitionIndex].d[DIR] < points[new_s].d[DIR])
				swap(points,new_s,partitionIndex);
		else 
			push(new_s,partitionIndex);

		if(partitionIndex + 1 == new_f && 
			points[new_f].d[DIR] < points[partitionIndex].d[DIR])
				swap(points,partitionIndex,new_f);
		else
			push(partitionIndex+1,new_f);
	}
	printf("%d = maxTop\n",maxTop);
}
void sortBy(V2 * points,s32 s,s32 f,s32 DIR)
{
	QsortBy(points,s,f,DIR);
	//BubbleSortBy(points,s,f,DIR);
}
s32  exchangeDIR(s32 DIR)
{
	if(DIR == 0)
		return 1;
	else return 0;
}
void findMinMaxXY(V2 * points,s32 s,s32 f,f32 &minX,f32 &maxX,f32 &minY,f32 &maxY)
{
	FOR(i,s,f+1)
	{
		if(points[i].x<minX) minX =points[i].x;
		if(points[i].x>maxX) maxX =points[i].x;
		if(points[i].y<minY) minY =points[i].y;
		if(points[i].y>maxY) maxY =points[i].y;
	}
}
void recursiveKdPrep(V2 * points, s32 s, s32 f, s32 DIR )
{
	if(f-s <=1)
		return;
	else
	{
		f32 minX = 1e6,minY= 1e6,maxX= -1e6, maxY= -1e6;
		findMinMaxXY(points,s,f,minX,maxX,minY,maxY);
		if(maxX-minX >maxY-minY)
			sortBy(points,s,f,0);
		else 
			sortBy(points,s,f,1);
	}
	s32 mpt = (s+f)/2;
	recursiveKdPrep(points,s,mpt,DIR);
	recursiveKdPrep(points,mpt,f,DIR);
}

void recursiveKdInsert(KdNode *tree,V2 *points,s32 s,s32 f)
{
	if(f-s <= 1)
		return; 
	else
	{
		s32 mpt = (s+f)/2;
		NodeType type;
		if(SQ(points[f].x-points[s].x) >SQ(points[f].y - points[s].y))//assuming sorted array
			type = LR;
		else type = UD;
		insertIntoKdTree(*tree,points[mpt],type);
		recursiveKdInsert(tree,points,s,mpt);
		recursiveKdInsert(tree,points,mpt,f);
	}
}

V2 points[20];
void testKdNode()
{
	KdNode *tree = KDNode(V(0,0),LR);
	tree->isFull = false;
	FOR(i,0,20)
	{
		V2 pt = V(rand()%10,rand()%10);
		points[i]= pt;
	}
	recursiveKdPrep(&points[0],0,20,0);
	recursiveKdInsert(tree,points,0,20);

	printVec(findClosestPointRec(*tree,V(7,5)));
}

void fillFullHeightMap(gridData gridD,gridInfo d)
{
	gridData gd = IgridData(d.width,d.height,sizeof(f32));
	
	s32 *heightMap = (s32*)gd.data;
	f32 wToGridWidthRatio = d.width/d.gridW;
	f32 hToGridHeightRatio = d.height/d.gridH;
	f32 w_2 = d.width*d.width;
	f32 h_2 = d.height*d.height;
	f32 valAtMin = 0.5f;
	for(s32 y= 0;y<d.height;y++)
	{
		for(s32 x= 0;x<d.width;x++)
		{
			f32 x_2 = x*x;
			f32 y_2 = y*y;
			
			f32 val = (x_2+y_2)/(w_2+h_2);
			V2 p = {x,y};
			V2 startCell = findCell(d,p);
			f32 h0,h1,h2,h3;
			
			h0 = getCellData_s32(startCell.x,startCell.y,gridD    );
			h1 = getCellData_s32(startCell.x+1,startCell.y,gridD  );
			h2 = getCellData_s32(startCell.x,startCell.y+1,gridD  );
			h3 = getCellData_s32(startCell.x+1,startCell.y+1,gridD);
			
			f32 tw = (p.x-(startCell.x*wToGridWidthRatio))/wToGridWidthRatio,
			th = (p.y-(startCell.y*hToGridHeightRatio))/hToGridHeightRatio;

			heightMap[y*d.width + x] = cosineInterpolation(bilerp(h0,h1,h2,h3,tw,th),valAtMin,val);
		}
	}
	writeBwBlockAsbmp(d.width,
					  d.height,
					  fillImgBasedOnNearestNeigboursResample(gd.data,0,d.width,0,d.height,d.width,d.height,750),
					  "img.bmp");
	writeBwBlockAsbmp(d.width
					 ,d.height,
					  gd.data,
					 "org_img.bmp");
	FgridData(gd);
}


void fillFullHeightMap(m4f32 gridD,gridInfo d)
{
	gridData gd = IgridData(d.width,d.height,sizeof(f32));
	
	f32 *heightMap = (f32*)gd.data;
	f32 wToGridWidthRatio = d.width/d.gridW;
	f32 hToGridHeightRatio = d.height/d.gridH;
	for(s32 y= 0;y<d.height;y++)
	{
		for(s32 x= 0;x<d.width;x++)
		{
			V2 p = {x,y};
			V2 startCell = findCell(d,p);
			f32 h0,h1,h2,h3;
			h0 = gridD.d[(s32)startCell.x][(s32)startCell.y];
			h1 = gridD.d[(s32)startCell.x+1][(s32)startCell.y];
			h2 = gridD.d[(s32)startCell.x][(s32)startCell.y+1];
			h3 = gridD.d[(s32)startCell.x+1][(s32)startCell.y+1];
			
			f32 tw = (p.x-(startCell.x*wToGridWidthRatio))/wToGridWidthRatio,
			th = (p.y-(startCell.y*hToGridHeightRatio))/hToGridHeightRatio;

			heightMap[y*d.width + x] = bilerp(h0,h1,h2,h3,tw,th);
		}
	}
	writeBwBlockAsbmp(d.width,d.height,gd.data,"img.bmp");
	FgridData(gd);
}



void testgridInfo()
{
	s32 sizeX = 1920,sizeY =1080;
	gridInfo d = {1920,1080,1000,1000};
	for(s32 i = 0; i<1000;i++)
	{
		V2 randP = V(rand()%sizeX, rand()%sizeY);
		V2 toTest = findCell(d,randP);
		int x =0 ;
	}
}


struct mat3
{
	f64 d[9];
	f64 operator [](s32 i)
	{
		return d[i];
	}
};

f64 det(f64 mat[9])
{
  f64 ret =0;
  
  ret += mat[0] * (mat[4]*mat[8] - mat[5]*mat[7]);
  ret -= mat[3] * (mat[1]*mat[8] - mat[2]*mat[7]);
  ret += mat[6] * (mat[1]*mat[5] - mat[2]*mat[4]);
  return ret;
}
mat3 addmat(mat3 a, mat3 b,f64 s)
{
	mat3 ret = {0};
	for(s32 i=0;i<3;i++)
	{
		for(s32 j=0;j<3;j++)
			ret.d[i*3 +j] = a[i*3+j]+(s*b[i*3+j]);
	}
	return ret;
}
void swaprows(f64 *mat,s32 t,s32 b)
{
	f64 v[3] = {mat[t],mat[t+1],mat[t+2]};
	mat[t]   = mat[b];
	mat[t+1] = mat[b+1];
	mat[t+2] = mat[b+2];
	mat[b] = v[0];
	mat[b+1] = v[1];
	mat[b+2] = v[2];
}
s32 findfirsttonotequalzero(f64 mat[9],s32 row)
{
	s32 ret  = -1;
	for(s32 i=row+3;i<9;)
	{
		if(mat[i]!=0)
		{
			return i/3;
		}
		i+=3;
	 }
		return ret;
}

s32 getswappairedup(f64 mat[9] ,s32 cellIndex)
{
	if(cellIndex == 8)
	{
		if (mat[cellIndex - 2 ]!=0 && mat[2] != 0)
			return 0;
		else if (mat[cellIndex-1] != 0 && mat[5]!=0)
			return 1;
	}
	if(cellIndex == 4)
	{
		if (mat[cellIndex - 1 ]!=0 && mat[1] != 0)
			return 0;
		else if (mat[cellIndex+1] != 0 && mat[7]!=0)
			return 2;
	}
}

void mulrow(f64 *mat ,s32 row,f64 val,s32 rowlen)
{
	for(s32 i = 0 ;i<rowlen; i++)
	{
		mat[row + i]*=val;
  }
}
void addRows(f64 *mat,s32 toAdd,s32 dest,s32 rowlen)
{
	for(s32 i = 0;i<rowlen;i++)
	{
		mat[dest+i] += mat[toAdd+i];
	}
}
void addrow(f64 *mat ,s32 toAdd ,s32 dest,f64 s,s32 rowlen)
{
  mulrow(mat,toAdd,s,rowlen);
  addRows(mat,toAdd,dest,rowlen);
  mulrow(mat,toAdd,1/s,rowlen);
}
void returnRowToOrder(f64 * mat,s32 cellIndex,f64 mulBy,s32 rowlen)
{
	mulrow(mat,(cellIndex/3)*3,mulBy,rowlen);
}
void nullifyValuesNotinPlace(f64 *mat,f64 *ret,s32 rowlen)
{  
  if (mat[3] != 0)
  {
	  f64 val = -mat[3];
      addrow(mat,0,3,val,rowlen); 
      addrow(ret,0,3,val,rowlen);
	  if(mat[4]<1e-6&&mat[4]>-1e-6)
	  {
		  if(mat[7]!=0)
		  {
			swaprows(mat,3,6);
			swaprows(ret,3,6);
			val = 1.f/mat[8];
			returnRowToOrder(mat,6,val,rowlen);
			returnRowToOrder(ret,6,val,rowlen);
			val = -mat[3];
			addrow(mat,0,3,val,rowlen); 
			addrow(ret,0,3,val,rowlen);
		  }

	  }
	  val = 1.f/mat[4];
	  returnRowToOrder(mat,4,val,rowlen);
	  returnRowToOrder(ret,4,val,rowlen);
  }
  if (mat[6] != 0)
  {
	  f64 val = -mat[6];
      addrow(mat,0,6,val,rowlen);
      addrow(ret,0,6,val,rowlen);
	  val = 1.f/mat[8];
	  returnRowToOrder(mat,8,val,rowlen);
	  returnRowToOrder(ret,8,val,rowlen);
  } 
  if (mat[1] != 0)
  {
	  f64 val = -mat[1];
      addrow(mat,3,0,val,rowlen); 
      addrow(ret,3,0,val,rowlen);
	  val = 1.f/mat[0];
	  returnRowToOrder(mat,0,val,rowlen);
	  returnRowToOrder(ret,0,val,rowlen);
  }
   if (mat[7] != 0)
  {
	  f64 val = -mat[7];
      addrow(mat,3,6,val,rowlen); 
      addrow(ret,3,6,val,rowlen);
	  val = 1.f/mat[8];
	  returnRowToOrder(mat,8,val,rowlen);
	  returnRowToOrder(ret,8,val,rowlen);
  } 
  if (mat[2] != 0)
  {
	  f64 val = -mat[2];
      addrow(mat,6,0,val,rowlen);
      addrow(ret,6,0,val,rowlen);
	  val = 1.f/mat[8];
	  returnRowToOrder(mat,8,val,rowlen);
	  returnRowToOrder(ret,8,val,rowlen);
  }
  if (mat[5] != 0)
  {
	  f64 val = -mat[5];
      addrow(mat,6,3,val,rowlen);
      addrow(ret,6,3,val,rowlen);
	  val = 1.f/mat[8];
	  returnRowToOrder(mat,8,val,rowlen);
	  returnRowToOrder(ret,8,val,rowlen);
  }
}

mat3 inverse(mat3 &mat,s32 rowlen=3)
{
	assert(det(mat.d)!=0);
	mat3 ret ={1,0,0,
               0,1,0,
		       0,0,1};

	for(s32 i = 0; i<3; i++)
	{
		if(mat.d[i*4]==0)
		{
			s32 toswapwith = findfirsttonotequalzero(mat.d,i*4);
			if(toswapwith == -1){

				toswapwith = getswappairedup(mat.d,i*4);
				swaprows(mat.d,i*3,toswapwith*3);
				swaprows(ret.d,i*3,toswapwith*3);
				i = -1; 
				continue;
			}
			swaprows(mat.d,i*3,toswapwith*3);
			swaprows(ret.d,i*3,toswapwith*3);
		}
		if(mat.d[i*4]!=0 && mat.d[i*4] != 1)
		{
			f64 val = 1.f/mat.d[i*4];
			mulrow(mat.d,i*3,val,rowlen);
			mulrow(ret.d,i*3,val,rowlen);
		}
	}
  nullifyValuesNotinPlace(mat.d,ret.d,rowlen);
  return ret;
}

void testQuadTree(QuadNode &head,s32 w,s32 h,s32 pointsCount)
{
	BB bb ={0,0,w,h};
	initQuadNode(head,bb);
	for(int i = 0; i<pointsCount;i++)
	{
		float toInsert[2] = {Arand()%w/3 +i%w/2 +1 +1,Arand()%h/3+ i%h/2+1};
		//float toInsert[2] = {((s32)Arand()%(int)((f32)w/1.4) + w/1.4)%w +1,(Arand()%(s32)((f32)h/1.4)+h/1.4)%h+1};		
		insertIntoTree(&head,toInsert,0);
	}
		
}

void DrawQuadTree(QuadNode * head,s32 w,s32 h,u8* pallate,RGB color)
{
	if(!head)
		return;
	RGB red = {255,0,0};
	RGB green= {0,255,0};
	bool hasChildren = false;
	for(int i = 0;i<4;i++)
	{
		if(head->Nodes[i])
		{
			hasChildren = true;
			if(head->Nodes[i]->hasData || head->Nodes[i]->currentTop==1)
			{
				
				CreatePallateWithALineOnIt(*(V2*)head->Nodes[i]->data[0],*(V2*)head->Nodes[i]->data[0],w,h,pallate,red);
			}
			{DrawAABB(head->Nodes[i]->bb,w,h,pallate,color);}
			DrawQuadTree(head->Nodes[i],w,h,pallate,color);
		}	
	}
	if(!hasChildren)
	{
		for(int i = 0; i<DEVIDE_QUADRANT_TRASHOLD;i++)
		{
			CreatePallateWithALineOnIt(*(V2*)head->data[i],*(V2*)head->data[i],w,h,pallate,red);
		}
	}
}
void testMedianSearch()
{
	s32 amount = 10000000;
	f32 * arr = (f32*) malloc(sizeof(float)*amount);
	FOR(i,0,amount)
	{
		arr[i] = Arand()%amount;
	}
	float median = findMedian(arr,0,amount-1,(amount)/2);
}
b8 testQuadTreeDraw(b8 continueFlow)
{
	//testMedianSearch();
	s32 w = 1000,h = 1000;
	srand(133);
	QuadNode head = {0};
	testQuadTree(head,w-1,h-1,20000);

	u8 * squereTest = 0;

	InitPallate(w,h,squereTest);
	DrawQuadTree(&head,w,h,squereTest,black);
	writeBlockAsbmp(w,h,squereTest,"squereTest.bmp");
	long t = time(0);
	writeBlockAsbmp(w,h,slowFillImgBasedOnNearestNeigboursResample(squereTest,0,w,0,h,w,h,100),"SlowSquereTest.bmp");
	printf("slow took :%d\n",time(0)-t);
	t = time(0);
	writeBlockAsbmp(w,h,fillImgBasedOnNearestNeigboursResample(squereTest,0,w,0,h,w,h,1000000),"squereTestQuadTreeResample.bmp");
	printf("fast took :%d\n",time(0)-t);
	
	return continueFlow;
}

void testQsort(s32 pointCount,s32 maxValue)
{
	V2 * points = (V2*)MyAlloc(pointCount*sizeof(V2));
	FOR(i,0,pointCount)
	{
		points[i] = V(Arand()%maxValue,Arand() % maxValue);
	}
	s32 DIR = 0;
	QsortBy(points,0,pointCount-1,DIR);
	drawHistogram(points,0,pointCount-1,1000.f/(pointCount+1),20,1000,DIR);	
}
void drawHistogram(V2 *points,s32 s ,s32 f,f32 widthPerBlock,f32 minY,f32 maxY,s32 DIR)
{
	f32 *distinctValues = (f32*)MyAlloc(sizeof(f32)*(f-s+1));
	s32 * countPerValue = (s32*)MyAlloc(sizeof(s32)*(f-s+1));
	memset(countPerValue,0,sizeof(s32)*(f-s+1));
	s32 valueCount = 0;
	s32 maxCount = (s32)-1e6;
	FOR(i,s,f+1)
	{
		f32 currentValue = points[i].d[DIR];
		b8 added = false;
		FOR(j,0,valueCount)
		{
			if(currentValue == distinctValues[j])
			{
				countPerValue[j]++;
				if(maxCount < countPerValue[j])
					maxCount = countPerValue[j];
				added = true;
				break;
			}
		}
		if(!added)
		{
			distinctValues[valueCount] = currentValue;
			countPerValue[valueCount++] = 1;
		}
	}
	u8 * Pallate;
	s32 w = 1000;
	s32 h = 1000;
	InitPallate(w,h,Pallate);
	widthPerBlock = (f32)w/(valueCount+1);
	widthPerBlock = widthPerBlock < 1 ?  1: widthPerBlock;
	FOR(i,0,valueCount)
	{
		BB bb = {i*widthPerBlock,
				 minY,
				 (i+1)*widthPerBlock -1,
				 maxY/(maxCount + 1 - countPerValue[i])};

		if(i%2 == 0)
			FillAABB(bb,w,h,Pallate,yellow);
		else 
			FillAABB(bb,w,h,Pallate,black);
	}
	writeBlockAsbmp(w,h,Pallate,"histogramTest.bmp");
}

#define BLUR_FILTER_COUNT 2
#define EDGE_DETECT_FILTER_COUNT 6
#define F_NAME(fileName,ext) fileName ext
image kernBlur[BLUR_FILTER_COUNT];
image kern5x5;
image kernEdgeDetect[EDGE_DETECT_FILTER_COUNT];
f32 divF = 1.f/9.f;
		f32 one_16 = 1.f/16;
		f32 one_256 = 1.f/256;
		f32 blur[][9] =    {{divF,divF,divF,
					          divF,divF,divF,
						       divF,divF,divF},
							   {one_16,one_16*2,one_16,
							   one_16*2,one_16*4,one_16*2,
							   one_16,one_16*2,one_16}};
		f32 sharpen[9] = {0,-1 ,0,
						 -1 ,5 ,-1,
					      0 ,-1 ,0};
		f32 gaussianBlur5x5[25] = {1,4 ,6 ,4 ,1,
							       4,16,24,16,4,
							       6,24,36,24,6,
								   4,16,24,16,4,
								   1,4 ,6 ,4 ,1};
			f32 edgeDetect[][9] = 
				{{1, 1, 1,
			      1, 0, -1,
			      -1, -1, -1},
				{1,1,1,
			     0,0, 0,
			    -1,-1, -1},
				{1, 0, -1,
			     1, 0, -1,
			     1, 0, -1},
				{-1, 0, 1,
			     -1, 0, 1,
			     -1, 0, 1},
				{1,0,1,
				 0,-4, 0,
				 1,0, 1},
				{-1,-1 ,-1,
			    -1 ,8 ,-1,
				-1 ,-1 ,-1}};
void initKernels()
{
		for(s32 i = 0 ; i< BLUR_FILTER_COUNT ; i++){kernBlur[i].w =3;kernBlur[i].h =3;kernBlur[i].fdata =blur[i];}

		for(s32 i = 0; i<25; i++) gaussianBlur5x5[i] *= one_256;
		kern5x5.w = 5; kern5x5.h = 5;
		kern5x5.fdata = gaussianBlur5x5;
		for(s32 i = 0 ; i< EDGE_DETECT_FILTER_COUNT ; i++) {kernEdgeDetect[i].w =3;kernEdgeDetect[i].h =3;kernEdgeDetect[i].fdata =edgeDetect[i];}
}
void testConvLayer()
{
	initKernels();
	image img0; 
	GetImage("halo.bmp",img0);
	image img = DownSampleRGB(img0,128,64);
	conv_layer l = ConvertImageToConvInput(img);
	//WriteImageAsbmp(ConvertConvLayerToImage(l,0),"lineTest2.bmp");
	//conv_layer * test = CreateConvLayer(&l,kernBlur,12,BLUR_FILTER_COUNT);
	V2 strides;
	strides.xi = strides.yi = 1;
	conv_layer * test = CreateConvLayer(&l,kernEdgeDetect,strides,EDGE_DETECT_FILTER_COUNT);
	conv_net network = {};
	AddLayerToNetwork(network,l);
	AddLayerToNetwork(network,*test);
	ComputeConvOutput(network);
	image output = ConvertConvLayerToImage(network.layers[1],0);
	//Treshold(output,7);

	WriteImageAsbmp(output,"halo2.bmp");
}
void testPerfStringCompare()
{
#define TYPE __m128i
	__declspec(align(16)) char A[] = "ababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcdababcdefghababcdzababcdefghababcd";
	__declspec(align(16)) char B[] = "ababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcdababcdefghababcdxababcdefghababcd";
	s32 lenA = ArrLen(A);
	s32 lenB = ArrLen(B);
	double resAsum=0,resBsum=0;
	s32 iterCount = 100;
	for(s32 t = 0 ;t<iterCount;t++)
	{
		StartCounterA(normalStrCompare);
		char * TA  = A;
		char * TB  = B;
		while((*(TA++)==*(TB++)));
		double resA = GetCounter(normalStrCompare);
		
		StartCounterA(paddedStringCompare);
		s32 aLenMod = ArrLen(A)% sizeof(TYPE);
		s32 bLenMod = ArrLen(B)% sizeof(TYPE);
		s32 lenAP = ArrLen(A)/sizeof(TYPE) + (aLenMod>0);
		s32 lenBP = ArrLen(B)/sizeof(TYPE) + (bLenMod>0);
		s32 minLenAB = min(lenAP,lenBP);
		TA = A;
		TB = B;
		s32 lenA0 = 0;
		s32 lenB0 = 0;

		b8 equal = true;

		__m128i zero = _mm_set_epi32(-1,-1,-1,-1);
		for(s32 c = 0; equal && c<minLenAB;c++)
		{
			s32 offset = c*sizeof(TYPE);
			#ifdef _INCLUDED_NMM
				__m128i A128 = _mm_stream_load_si128 ( (__m128i *)A) ;
				__m128i B128 = _mm_stream_load_si128 ( (__m128i *)B) ;
				__m128i resCmp128 = _mm_cmpeq_epi8(A128,B128);
				if(!_mm_testz_si128(resCmp128,zero))
					equal = false;
			#else
			if(*(TYPE*)(A + offset) ^ *(TYPE*)(B + offset))
				equal = false;
			#endif
			
		}
		double resB = GetCounter(paddedStringCompare);
		
		resAsum += resA;
		resBsum += resB;
	}
	
	printf("%.10f, %.10f %.10f\n",resBsum/iterCount,resAsum/iterCount, resAsum/resBsum);
}
void testHouseDraw()
{
	f32 lengths[]={2.88,.13,.11,2.26,.20,.10,.11,.13,.09,.23,.10,2.85,2.99,4.1,.10,.03,.97,2.33,.02,.21,.14,.93,1.11,.74,.10,.70,1.83,1.12,.13,.64,.10,.65,.91,2.61,.10,2.60,2.80,2.43,1.92,.10,1.93,1.41,1.93,.67,.11,.75,2.28,2.38};
	directions dirs[] = {LEFT,LEFT,RIGHT,LEFT,LEFT,RIGHT,RIGHT,LEFT,RIGHT,LEFT,RIGHT,RIGHT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT,RIGHT,LEFT,RIGHT,RIGHT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT,LEFT,RIGHT,RIGHT,RIGHT,LEFT};
	f32 lengths0[] = {2.07,0.04,0.19,.17,.19,.03,1.96,.07,.11,.16};
	//directions dirs0[]= {LEFT,RIGHT,LEFT,LEFT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT};
	directions dirs0[]= {LEFT,RIGHT,LEFT,LEFT,LEFT,LEFT,RIGHT,RIGHT,LEFT,LEFT,LEFT};
	line_direction * test  = CreateLineDirection(lengths,dirs,ArrLen(lengths),HORIZONTAL,V(1900,1900),200);
	line_direction * test0  = CreateLineDirection(lengths0,dirs0,ArrLen(lengths0),HORIZONTAL,V(1096,911),200);
	image img0={2000,2000};
		srand(time(0));
		
		
		CreatePerpPolygon(test,ArrLen(dirs),img0.udata,img0.w,img0.h);
		CreatePerpPolygon(test0,ArrLen(dirs0),img0.udata,img0.w,img0.h);
		char numberstr[256];
		V2 startPos = {300,900};
		FOR(i,0,ArrLen(lengths))
		{
			if(startPos.x > img0.w || startPos.y >img0.h)
				break;
			sprintf(numberstr,"%d",(s32)(100*lengths[i]));
			WriteNumber(numberstr,strlen(numberstr),img0.udata,img0.w,img0.h,startPos,0.1);
			startPos.y += 23;
		}
		writeBlockAsbmp(img0.w,img0.h,img0.udata,"houseSketch.bmp");
}
void testSimpleConv()
{
	#define RANODM_FILTER_COUNT 3
		f32 random[RANODM_FILTER_COUNT][9];
		image kernRand[RANODM_FILTER_COUNT];
		for(s32 j =0 ;j<RANODM_FILTER_COUNT;j++)
		{
			kernRand[j].h = kernRand[j].w = 3;
			
			s32 kernSize = kernRand[j].w * kernRand[j].h;
			for (s32 i = 0; i <kernSize ;i++)
			{
				random[j][i] = (f32)(rand())/0xffff;
			}
			kernRand[j].fdata = random[j];
		}
		
		image img,imgA;
		#define FILE_NAME "img"
		GetImage( F_NAME(FILE_NAME,".bmp"),img);
		image smallerVer = DownSampleRGB(img,64,64);
		image t = smallerVer;
		char fileName[256];
		
		for(s32 i=0; i < RANODM_FILTER_COUNT; i++)
		{
			image t0 = ConvR(t,kernRand[i]);
			sprintf(fileName,"%s%d.bmp\0",FILE_NAME,i+1);
			writeBlockAsbmp(t0.w,t0.h,t0.udata,fileName);
			free(t0.udata);
		}
		//t = ConvR(t,kernEdgeDetect[0]);
		//Treshold(t,10);
		//writeBlockAsbmp(t.w,t.h,t.udata, F_NAME(FILE_NAME,"1.bmp"));
		s32 count = 0;
		for(s32 i = 0; i<count; i++)
		{
			image afterPass = Conv(img,kern5x5);
			image afterDiff = Combine(img,afterPass,SUB);
			image afterAdd  = Combine(img,afterDiff,ADD);
			image afterConvAdd = Conv(afterAdd,kern5x5);

			free(img.udata);
			//free(afterDiff.udata);
			//free(afterAdd.udata);
			//img.w = afterAdd.w;
			//img.h = afterAdd.h;
			img.udata=afterConvAdd.udata;
			//Invert(img);
		}
		//Invert(img);
		
		//kern.fdata = downSample;
		//u8 * conved = Conv(img,kern).udata;
		//img.udata = conved;
		
		writeBlockAsbmp(img.w,img.h,DownSampleRGB(img,30,30).udata,"halo1.bmp");

		for(s32 j =0 ;j<RANODM_FILTER_COUNT;j++)
		{
			kernRand[j].h = kernRand[j].w = 3;
			
			s32 kernSize = kernRand[j].w * kernRand[j].h;
			for (s32 i = 0; i <kernSize ;i++)
			{
				random[j][i] = (f32)(rand())/0xffff;
			}
			kernRand[j].fdata = random[j];
		}
		
		GetImage( F_NAME(FILE_NAME,".bmp"),img);
		smallerVer = DownSampleRGB(img,64,64);
		t = smallerVer;
		
		for(s32 i=0; i < RANODM_FILTER_COUNT; i++)
		{
			image t0 = ConvR(t,kernRand[i]);
			sprintf(fileName,"%s%d.bmp\0",FILE_NAME,i+1);
			writeBlockAsbmp(t0.w,t0.h,t0.udata,fileName);
			free(t0.udata);
		}
		//t = ConvR(t,kernEdgeDetect[0]);
		//Treshold(t,10);
		//writeBlockAsbmp(t.w,t.h,t.udata, F_NAME(FILE_NAME,"1.bmp"));
		
		count = 0;
		for(s32 i = 0; i<count; i++)
		{
			image afterPass = Conv(img,kern5x5);
			image afterDiff = Combine(img,afterPass,SUB);
			image afterAdd  = Combine(img,afterDiff,ADD);
			image afterConvAdd = Conv(afterAdd,kern5x5);

			free(img.udata);
			//free(afterDiff.udata);
			//free(afterAdd.udata);
			//img.w = afterAdd.w;
			//img.h = afterAdd.h;
			img.udata=afterConvAdd.udata;
			//Invert(img);
		}
		//Invert(img);
		
		//kern.fdata = downSample;
		//u8 * conved = Conv(img,kern).udata;
		//img.udata = conved;
		
		writeBlockAsbmp(img.w,img.h,DownSampleRGB(img,30,30).udata,"halo1.bmp");

}
void DrawPointRGB(V2 a,image &img,RGB color)
{
	s32 index = a.y * img.w + a.x;
	assert((img.w * img.h-1) > index);
	img.rgbdata[index] = color;
}
void testCubicCurve()
{
	{
	image img = CreateImageRGB(1000,1000);
	for(s32 i = 0;i<img.h;i++)
		for(s32 j = 0; j<img.w;j++)
		{
			s32 index = i*img.w + j;
			img.rgbdata[index] = white;
		}
	f32 res = .05;
	V2 points [100];
	s32 arrayLength = ArrLen(points);
	s32 ratioWAL = (img.w/arrayLength);
	FOR(s,1,arrayLength)
	{
		points[s].x = (arrayLength - s)*(img.w/arrayLength);
		points[s].y = cos(points[s].x*s)*20 + sin((f32)(s*points[s].x))*210 +500 ;
	}
	


	for(f32 t =0; t <= 1.f; t+=res)
	{
		for(s32 p = 0 ; p<ArrLen(points); p+=3)
		{
			if(p+2>=ArrLen(points))
				break;
			CreatePallateWithALineOnIt(cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t),
				cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res),
				img.w,img.h,img.udata,blue);
			if(p+4<= arrayLength-1)
				CreatePallateWithALineOnIt(.5f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t)),
					.5f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t+res)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res)),
					img.w,img.h,img.udata,red);
			if(p+5<= arrayLength-1)
				CreatePallateWithALineOnIt(.333333f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t)),
					.333333f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t+res)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t+res)),
					img.w,img.h,img.udata,green);
			if(p+6<= arrayLength-1)
				CreatePallateWithALineOnIt(.25f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t)),
					.25f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t+res)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t+res)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t+res)),
					img.w,img.h,img.udata,black);
			if(p+7<= arrayLength-1){
				CreatePallateWithALineOnIt(.125f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t))+.1f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t)+cubicCurvePoint2(points[p+4],points[p+5],points[p+6],points[p+7],t)),
					.125f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t+res)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t+res)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t+res))+.1f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t+res)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t+res)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t+res)+cubicCurvePoint2(points[p+4],points[p+5],points[p+6],points[p+7],t+res)),
					img.w,img.h,img.udata,yellow);
				CreatePallateWithALineOnIt(.2f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t)+cubicCurvePoint2(points[p+4],points[p+5],points[p+6],points[p+7],t)),
					.2f*(cubicCurvePoint2(points[p+1],points[p+2],points[p+3],points[p+4],t+res)+cubicCurvePoint2(points[p],points[p+1],points[p+2],points[p+3],t+res)+cubicCurvePoint2(points[p+2],points[p+3],points[p+4],points[p+5],t+res)+cubicCurvePoint2(points[p+3],points[p+4],points[p+5],points[p+6],t+res)+cubicCurvePoint2(points[p+4],points[p+5],points[p+6],points[p+7],t+res)),
					img.w,img.h,img.udata,black);

			}


			//CreatePallateWithALineOnIt(points[p],
			//	points[p+1],
			//	img.w,img.h,img.udata,blue);
			//CreatePallateWithALineOnIt(points[p+1],
			//	points[p+2],
			//	img.w,img.h,img.udata,blue);
			//CreatePallateWithALineOnIt(points[p+2],
			//	points[p+3],
			//	img.w,img.h,img.udata,blue);
		}
	}
	WriteImageAsbmp(img,"cubicCurveTest.bmp");
}

}
void testQuadraticCurve()
{
	image img = CreateImageRGB(1000,1000);
	for(s32 i = 0;i<img.h;i++)
		for(s32 j = 0; j<img.w;j++)
		{
			s32 index = i*img.w + j;
			img.rgbdata[index] = white;
		}
	f32 res = .01;
	V2 points[200];
	FOR(s,0,200)
	{
		s32 maxValue = 100;
		f32 randAdd = Arand()%maxValue;
		randAdd +=  - maxValue/2;
		
		points[s].x =5*s;
		points[s].y =500 +200*cos((.1*s*randAdd))- randAdd;
		printf("%f\n",points[s].y);
	}


	for(f32 t =0; t < 1.; t+=res)
	{
		for(s32 p = 0 ; p<ArrLen(points); p+=2)
		{
			if(p+2>=ArrLen(points))
				break;
			CreatePallateWithALineOnIt(quadraticCurvePoint(points[p],points[p+1],points[p+2],t),
				quadraticCurvePoint(points[p],points[p+1],points[p+2],t+res),
				img.w,img.h,img.udata,black);
		}
	}
	WriteImageAsbmp(img,"quadraticCurveTest.bmp");
}
int main ()
{
	//testPerfStringCompare();
	//testQuadraticCurve();
	testCubicCurve();
	return 0;
	testConvLayer();
	return 0;

		s32 countPerHiddenLayer[] = {10};
		f32 input[] = {0,0,1,0,0,1,1,1};
		f32 target[] = {1,1,1,0};
		s32 inputCount = 2;
		s32 outputCount = 1;
		neural_net net = CreateNetwork(inputCount , ArrLen(countPerHiddenLayer) , countPerHiddenLayer,outputCount);
		//ProduceOutput(net);
		f32 error = 0;
		while(true){

			for(s32 i = 0; i<4;i++)
			{
				for(s32 j = 0;j<10;j++)
				{
					FillInput(net,&input[i*2],inputCount);
					ProduceOutput(net);
					BackPropFixValue(net,&target[i],0.01f);
				}
				error += (target[i] - net.output.output[0])/4;
			}
			printf("%f\n",error);
			
			if(bw(error,-1e-10,1e-10))
			{
				break;
			}
			error = 0;
		}
		return 0;
	

	s32 w = 1000,h = 1000;
	srand(133);

	//if(!testQuadTreeDraw(false))
	//	return;
	//testKdNode();
	//return 0 ;
	testQsort(1000,500);
	return 0;
	//testgridInfo();
	//testProjectionUnProjection();
	
#define PAIR(n) n,n

	u8* LineDrawTest = 0;
	
	V2 A =  V(0,0);
	V2 B= V(500,750);
	V2 C= B+ V(200,-300);
	//V2 D= B + V(300,-500);
	V2 D = V(999,999);
	//CreatePallateWithALineOnIt(A,B,w,h,LineDrawTest);
	//CreatePallateWithALineOnIt(B,C,w,h,LineDrawTest);
	//CreatePallateWithALineOnIt(C,D,w,h,LineDrawTest);
	V2 currentPoint = A;
	f32 precision = 200;
	//FOR(k,0,10)
	//{
	//	V2 A =  V(rand()%w,rand()%h);
	//	V2 B =  V(rand()%w,rand()%h);
	//	V2 C =  V(rand()%w,rand()%h);
	//	V2 D =  V(rand()%w,rand()%h);
	//	RGBA color = {rand()%255,rand()%255,rand()%255,1};
	//	V2 currentPoint = A;
	//	//for(f64 t = 0 ;t<1.f;t+=1.f/precision)
	//	//{
	//	//	V2 bezierControlPoint = bezierCurvePoint(A,B,C,D,1.f-t);
	//	//	CreatePallateWithALineOnIt(currentPoint,bezierControlPoint,w,h,LineDrawTest,color);
	//	//	currentPoint = bezierControlPoint;
	//	//}
	//	DrawTriangle(A,B,C,w,h,LineDrawTest,color);
	//}
		//A =  V(rand()%w,rand()%h);
		//B =  V(rand()%w,rand()%h);
		//C =  V(rand()%w,rand()%h);

		A = V(0,300);
		B = V(w,300);
		C = V(w/2,600);
		//RGBA color = {rand()%255,rand()%255,rand()%255,1};
		RGB color = {255,128,128};
		//DrawSirpensky(A,B,0,w,h,LineDrawTest,color);
		Z z = {0,0};
		DrawMendelbrotSet(z,1.5,20,w,h,LineDrawTest,color);
	if(LineDrawTest)
	{
		writeBlockAsbmp(w,h,LineDrawTest,"lineTest.bmp");
		//writeBwBlockAsbmp(w,h,fillImgBasedOnNearestNeigboursResample(LineDrawTest,0,w,299,600,w,h,300e3),"lineResample.bmp");
	}
	free(LineDrawTest);
	return 0 ;
	u8* RegTest = createRegressionTest(w,h,500);
	writeBwBlockAsbmp(w,h,RegTest,"org_regressionTest.bmp");
	writeBwBlockAsbmp(w,h,fillImgBasedOnNearestNeigboursResample(RegTest,0,w,0,h,w,h,5000),"regressionTest.bmp");
	free(RegTest);
	return 0;
	u8*sphereData =produceSphereHeightImg(w,h);
	writeBwBlockAsbmp(w,h,sphereData,"org_sphere.bmp");

	writeBwBlockAsbmp(w,h,
		fillImgBasedOnNearestNeigboursResample(sphereData,0,w,0,h,w,h,500),"sphere.bmp");
	free(sphereData);
	//gridInfo g = {1000,1000,PAIR(6)};
	//fillFullHeightMap(generateRandomGrid(0,255,g),g);
	return 0;
	for(s32 i = 0; i<100; i++)
	{
		mat3 matrix = {rand()%100,rand()%100,rand()%100,
					   rand()%100,rand()%100,rand()%100,
					   rand()%100,rand()%100,rand()%100};
		mat3 matBack = matrix;
		mat3 res= inverse(matrix);
		mat3 mat= inverse(res);
		res = addmat(matBack,mat,-1);
		int x= 0;

	}
	//mat3 matrix ={7,2,1,
	//			  0,3,-1,
	//			  -3,4,-2};

	//mat3 matrix = 
	//{
	//	6,-25,154,
	//	73,0,72,
	//	82,51,0,
	//};
	//mat3 matrix = 
	//{
	//	6,-2,4,
	//	73,2,1,
	//	82,3,0,
	//};
	mat3 matrix = 
	{
		0,3,6,
		1,2,0,
		22,3,0,
	};

	//mat3 matrix = 
	//{
	//	1,5,4,
	//	7,0,7,
	//	8,5,0,
	//};
	//mat3 matrix ={  1,2,3,
	//				0,1,4,
	//				5,6,0};
	
	//mat3 matrix ={  3,0,2,
	//				2,0,-2,
	//				0,1,1};
	mat3 result = inverse(matrix);
	matrix = inverse(result);

}
void writeBwBlockAsbmp(s32 w,s32 h,u8* data,char * fileName)
{
FILE *f;
int filesize = 54 + 3*w*h; //header + data size;

u8 *img =(u8*) malloc(3*w*h);
for(int i=0; i<h; i++)
{
    for(int j=0; j<w; j++)
	{
		s32 y=(h-1)-i, x=j;
		
		img[(x+y*w)*3+2] = data[(i*w+j)*sizeof(s32)];
		img[(x+y*w)*3+1] = data[(i*w+j)*sizeof(s32)];
		img[(x+y*w)*3+0] = data[(i*w+j)*sizeof(s32)];
	}
}

unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
unsigned char bmppad[3] = {0,0,0};

bmpfileheader[ 2] = (unsigned char)(filesize    );
bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
bmpfileheader[ 4] = (unsigned char)(filesize>>16);
bmpfileheader[ 5] = (unsigned char)(filesize>>24);

bmpinfoheader[ 4] = (unsigned char)(       w    );
bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
bmpinfoheader[ 6] = (unsigned char)(       w>>16);
bmpinfoheader[ 7] = (unsigned char)(       w>>24);
bmpinfoheader[ 8] = (unsigned char)(       h    );
bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
bmpinfoheader[10] = (unsigned char)(       h>>16);
bmpinfoheader[11] = (unsigned char)(       h>>24);

f = fopen(fileName,"wb");
fwrite(bmpfileheader,1,14,f);
fwrite(bmpinfoheader,1,40,f);
for(s32 i=0; i<h; i++)
{
    fwrite(img+(w*(h-i-1)*3),3,w,f);
    fwrite(bmppad,1,w%4,f);
}
fclose(f);
}
void writeBlockAsbmp(s32 w,s32 h,u8* data,char * fileName)
{
FILE *f;
int padSize  = w%4;
int sizeData = h * (w*3 + padSize);
int filesize = 54 + sizeData; //header + data size;

unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
unsigned char bmppad[3] = {0,0,0};

bmpfileheader[ 2] = (unsigned char)(filesize    );
bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
bmpfileheader[ 4] = (unsigned char)(filesize>>16);
bmpfileheader[ 5] = (unsigned char)(filesize>>24);

bmpinfoheader[ 4] = (unsigned char)(       w    );
bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
bmpinfoheader[ 6] = (unsigned char)(       w>>16);
bmpinfoheader[ 7] = (unsigned char)(       w>>24);
bmpinfoheader[ 8] = (unsigned char)(       h    );
bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
bmpinfoheader[10] = (unsigned char)(       h>>16);
bmpinfoheader[11] = (unsigned char)(       h>>24);



bmpinfoheader[20] = (unsigned char)(sizeData);
bmpinfoheader[21] = (unsigned char)(sizeData>>8);
bmpinfoheader[22] = (unsigned char)(sizeData>>16);
bmpinfoheader[23] = (unsigned char)(sizeData>>24);

f = fopen(fileName,"wb");
fwrite(bmpfileheader,1,14,f);
fwrite(bmpinfoheader,1,40,f);
s32 wT3 = w*3;
for(s32 i=0; i<h; i++)
{
    fwrite(data+(wT3*i),3,w,f);
	fwrite(bmppad,1,padSize,f);
}
fclose(f);
}
void WriteImageAsbmp(image &img, char * fileName)
{
	writeBlockAsbmp(img.w,img.h,img.udata,fileName);
}

f32* generateRandomPairsOfFloat(s32 num,s32 minX,s32 maxX,s32 minY,s32 maxY)
{
	f32 *ret = (f32*)malloc(2*num*sizeof(f32));

	FOR(i,0,2*num)
	{
		ret[i] =   clamp(Arand()%maxX  +  minX,maxX);
		ret[i+1] = clamp(Arand()%maxY  +  minY,maxY);
		i++;
	}
	return ret;
}
QuadNode* generateRandomQuadNode(s32 num,s32 minX,s32 maxX,s32 minY,s32 maxY)
{
	QuadNode *ret = (QuadNode*)malloc(sizeof(QuadNode));
	BB bb = {minX,minY,
		     maxX,maxY};
	initQuadNode(*ret,bb);	
	StartCounterA(StartTime);
	FOR(i,0,num)
	{
		f32 item[2] = {clamp(rand()%maxX  +  minX,maxX),
					   clamp(rand()%maxY  +  minY,maxY)};		
		insertIntoTree(ret,item);
	}
	double counter = GetCounter(StartTime);
	printf("insertion took : %.5f\n",counter);
	printf("Per Insertion: %.5f\n",counter/num);
	return ret;
}
float* rep(QuadNode *qn)
{	
	if(qn->currentTop>0)
	{	
		float *res = (float*)malloc(sizeof(f32)*2);
		res[0] = (qn->dataSum[0])/qn->currentTop;
		res[1] = (qn->dataSum[1])/qn->currentTop;
		return res;
	}
	else 	
		return 0;	
}
void getRepArray(QuadNode * CurrentNode,float items[4][2])
{
	items[0][0] = CurrentNode->Nodes[0]->dataSum[0]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[0][1] = CurrentNode->Nodes[0]->dataSum[1]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[1][0] = CurrentNode->Nodes[1]->dataSum[0]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[1][1] = CurrentNode->Nodes[1]->dataSum[1]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[2][0] = CurrentNode->Nodes[2]->dataSum[0]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[2][1] = CurrentNode->Nodes[2]->dataSum[1]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[3][0] = CurrentNode->Nodes[3]->dataSum[0]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
	items[3][1] = CurrentNode->Nodes[3]->dataSum[1]/(CurrentNode->currentTop == 0 ? 1e-10:CurrentNode->currentTop);
}
u8* fillImgBasedOnNearestNeigboursResample(u8*data,s32 minX,s32 maxX,s32 minY,s32 maxY,s32 w,s32 h,s32 neighboursNum)
{
	QuadNode * randomData = generateRandomQuadNode(neighboursNum,minX,maxX,minY,maxY);	
	u8* ret = (u8*)malloc(sizeof(s32)*w*h);
	__int64 counter;
	StartCounter(counter);
	FOR(i,minY,maxY)
	{
		FOR(j,minX,maxX)
		{
			s32 index = (i*w+j)*sizeof(s32);
			QuadNode *CurrentNode = randomData;
			s64 whileLoopCounter;
			
			while(CurrentNode->Nodes[0] &&
				  CurrentNode->Nodes[1] &&
				  CurrentNode->Nodes[2] &&
				  CurrentNode->Nodes[3])
			{
				s64 repCounter = 0;
				StartCounter(repCounter);
				float items[4][2];
				getRepArray(CurrentNode,items);
				
				//printf("repCounter: %.10f\n",GetCounter(repCounter));
				f32 minDist =2e6;
				QuadNode *TempNode = 0;
				s64 iiLoopCounter;
				StartCounter(iiLoopCounter);
				FOR(ii,0,4)
				{
					if(items[ii])
					{	
						f32 dx = items[ii][0] - j,
							dy = items[ii][1] - i;

						f32 dist = dx*dx + dy*dy;
						if(dist<minDist)
						{
							minDist = dist;
							TempNode = CurrentNode->Nodes[ii];
						}
					}
				}
				CurrentNode = TempNode;
				//printf("iiloopCounter: %.5f\n",GetCounter(iiLoopCounter));

			}
			StartCounter(whileLoopCounter);
			
			f32 currentMinDist = 1e7;
			s32 bestIndex = -1;
			FOR(k,0,CurrentNode->currentTop)
			{
				f32 dx = CurrentNode->data[k][0] - j,
					dy = CurrentNode->data[k][1] - i;
				f32 dist = dx*dx + dy*dy;
				if(currentMinDist>dist)
				{
					currentMinDist = dist;
					bestIndex = k;
				}
			}
			if(bestIndex!=-1)
			{
				ret[index] = data[(s32)(CurrentNode->data[bestIndex][0] + CurrentNode->data[bestIndex][1]*w)*sizeof(s32)];
				ret[index+1] = data[(s32)(CurrentNode->data[bestIndex][0] + CurrentNode->data[bestIndex][1]*w)*sizeof(s32)+1];
				ret[index+2] = data[(s32)(CurrentNode->data[bestIndex][0] + CurrentNode->data[bestIndex][1]*w)*sizeof(s32)+2];
			}
			//printf("whileLoopCounter: %.10f\n",GetCounter(whileLoopCounter));
			
			//printf("findTheClosestNeighboorTook: %.5f\n",GetCounter(counter));
		}
	}
	double ttl = GetCounter(counter);
	printf("PixelLoopTook: %.5f\n",ttl);
	printf("CountPerPixel: %.5f\n",ttl/((maxX-minX)*(maxY-minY)));
	free(randomData);
	return ret;
}

u8* slowFillImgBasedOnNearestNeigboursResample(u8*data,s32 minX,s32 maxX,s32 minY,s32 maxY,s32 w,s32 h,s32 neighboursNum)
{
	f32 * randomData = generateRandomPairsOfFloat(neighboursNum,minX,maxX,minY,maxY);
	u8* ret = (u8*)malloc(sizeof(s32)*w*h);
	FOR(i,minY,maxY)
	{
		FOR(j,minX,maxX)
		{
			__int64 counter;
			StartCounter(counter);
			s32 index = (i*w+j)*sizeof(s32);
			s32 closestXIndex =-1;
			f32 minDist = 10e7;
			FOR(k,0,neighboursNum*2)
			{
				f32 dx = randomData[k]-j;
				f32 dy = randomData[k+1]-i;
				f32 dist = dx*dx + dy*dy;
				if(dist<minDist)
				{
					minDist = dist;
					closestXIndex = k;
				}
				k=k+1;
			}
			if(closestXIndex>=0)
			{
				ret[index] = data[(s32)(randomData[closestXIndex] + randomData[closestXIndex+1]*w)*sizeof(s32)]; 
				ret[index+1] = data[(s32)(randomData[closestXIndex] + randomData[closestXIndex+1]*w)*sizeof(s32)+1]; 
				ret[index+2] = data[(s32)(randomData[closestXIndex] + randomData[closestXIndex+1]*w)*sizeof(s32)+2]; 
			}
			else ret[index] = 0 ;
			//printf("insertionTook: %.5f\n",GetCounter(counter));
		}
	}
	free(randomData);
	return ret;
}




u8* produceSphereHeightImg(s32 width,s32 height)
{
	f64 incI = 1.f/width;
	f64 incJ = 1.f/height;
	f32 sqrt_2 = sqrt(2.f);
	s32 maxPixelValue = 2*256;
	f32 maxValue =-1;
	f32 minValue = 1e6;
	s32* data =  (s32*) malloc(width*height*sizeof(s32));
	for(f64 i=0; i<=1;i+= incI)
	{
		for(f64 j=0; j<=1;j+=incJ)
		{
			s32 x = i*width;
			s32 y = j*height;
			f32 i_2 = SQ(i-0.5);
			f32 j_2 = SQ(j-0.5);
			u8 pixelValue = sqrt(.25f -(i_2+j_2))*maxPixelValue;
			data[y*width +x] = pixelValue;
		}
	}
	//printf("maxValue : %3f ,minValue : %3f\n",maxValue,minValue);
	return (u8*)data;
}
V2 * generateRandomPoints2D(s32 count,f32 minX,f32 maxX,f32 minY,f32 maxY)
{
	V2 * ret = (V2 *) malloc(count*sizeof(V2));
	FOR(i,0,count)
	{
		s32 x = minX + rand()%(s32)(maxX-minX);
		s32 y = minY + rand()%(s32)(maxY-minY);
		ret[i] = V( x, y);
	}
	return ret;
}

void fillRegressionPallete(V2*pts,s32 point_count,s32* data,s32 w, s32 h)
{
	memset(data,0,w*h*sizeof(s32));
	V2 sum = {0};
	FOR(P,0,point_count)
	{
		data[(s32)(pts[P].y*w + pts[P].x)] = 255;
		sum += pts[P];
	}
	f32 slope = sum.y/sum.x;
	FOR(i,0,w)
	{   s32 x = i;
		s32 y = slope*x;
		s32 index = y*w +x;
		if(index < w*h)
			data[index] = 255;
		else break;
	}
	printf("%3f, %3f, %3f \n",sum.x,sum.y ,slope);
}
u8 * createRegressionTest(s32 w,s32 h,s32 point_count)
{
	f32 minX = 0,maxX = w;
	f32 minY = 0,maxY = h;
	V2 * randPoints = generateRandomPoints2D(point_count,minX,maxX,minY,maxY);

	s32 * pallate = (s32 *) malloc(w*h*sizeof(s32));

	fillRegressionPallete(randPoints,point_count,pallate,w,h);
	return (u8*) pallate;
}

V2 bezierCurvePoint(V2 a, V2 b ,V2 c ,V2 d,f32 t)
{
	V2 ret = (t*t*t*a) + (3*(1.f-t)*t*t*b) +3*SQ(1.f-t)*t*c + SQ(1.f-t)*(1.f-t)*d;
	return ret;
}
V2 quadraticCurvePoint(V2 a ,V2 b ,V2 c, f32 t)
{
	V2 apb = a+b;
	V2 coff0 = 2*((c-a) -2*(b-a));
	V2 coff1 = 2*(b-a) -.5*coff0;
	V2 ret = (t*t)*coff0 + t*coff1 + a;
	return ret;
}

V2 cubicCurvePoint(V2 a ,V2 b ,V2 c,V2 d ,f32 t)
{
	V2 apb = a+b;
	f32 third = 1.f/3;
	f32 thirdCubed = 1.f/27;
	f32 twoThirdCubed = 8.f/27;
	f32 thirdSqured = 1.f/9;
	f32 twoThirdSqured = 4.f/9;
	f32 twoThird = 2.f/3;
	V2 res2 =  (1.f/(thirdSqured-thirdCubed))*((b-a) - thirdCubed*(d-a));
	V2 res3 =  (1.f/(twoThird - twoThirdCubed))*(c-a - twoThirdCubed*(d-a));
	f32 A = (twoThirdSqured - twoThirdCubed)/(twoThird - twoThirdCubed);
	f32 B = (third -thirdCubed)/(thirdSqured-thirdCubed);

	V2 coff2 = res3 - A*res2;
	V2 coff1 = res2 -B*(res3-A*res2);
	V2 coff0 = c-a - coff1 - coff2;
	V2 ret = (t*t*t)* coff0 + (t*t)*coff1 + t*coff2 + a;
	return ret;
}

V2 cubicCurvePoint2(V2 a ,V2 b ,V2 c,V2 d ,f32 t)
{	V2 A = d-a,B = b-a,C = c-a;
	f32 third = 1.f/3;
	f32 thirdCubed = 1.f/27;
	f32 twoThirdCubed = 8.f/27;
	f32 thirdSquered = 1.f/9;
	f32 twoThirdSquered = 4.f/9;
	f32 twoThird = 2.f/3;
	V2 I = C - (twoThirdCubed*A),H = B-thirdCubed*A;

	f32 D = -thirdCubed + thirdSquered,E =-thirdCubed+third,
		F = -twoThirdCubed + twoThirdSquered ,G = -twoThirdCubed +twoThird;
	f32 K = 1.f-(E/D)*(F/G);
	V2 J = (1.f/G)*I - ((F/G)*(1/D)*H),L = (1.f/D)*H - E/D*((1.f/K)* J );


	V2 coffA = A-(1.f/K)*J -L;
	V2 coffB = L;
	V2 coffC = (1.f/K)*J;
	//V2 coffB =-(270/12)*(8.f/270)*(27*C-8*A) + B - (1.f/27)*A;
	//V2 coffC =  (1.f/10.f)*27*C - 8*A - (4.f/10) * coffB;
	//V2 coffA = A - coffB -coffC;


	

	V2 ret = (t*t*t)*coffA + (t*t)*coffB + t* coffC + a;

	return ret;

}

void InitPallate(f32 w, f32 h,u8* &pallate,s32 value)
{
	pallate = (u8*)malloc(w*h*sizeof(s32));
	memset(pallate,value,w*h*sizeof(s32));
}
void CreatePallateWithALineOnIt(V2 start,V2 end,f32 w,f32 h,u8*& pallate,RGB color)
{
	{//flip RGB to BGR
		u8 t0 = color.r;
		color.r = color.b;
		color.b = t0;
	}

	if(!pallate)
		InitPallate(w,h,pallate);
	
	V2 Diff = end - start;
	RGB * data =(RGB*)pallate;
	f32 eps =1e-2;
	if(bw(Diff.y,-eps,eps) && bw(Diff.x,-eps,eps))
	{
		V2 currentPoint = start;
		s32 indT = (s32)currentPoint.y*w + (s32)currentPoint.x;
		if(bw(indT,0,w*h))
			data[indT] = color;
		return;
	}
	f64 dist = sqrt(dot(Diff,Diff));
	f64 addition = 1./dist;
	V2 prevPoint = start;

	if(Diff.x == 0 || Diff.y == 0)
	{
		s32 DIR = Diff.x == 0 ? 1:0;
		s32 dir = (Diff.d[DIR]<0)?-1:1;

		for(s32 i = start.d[DIR];
			dir>0 ? i<=end.d[DIR]: i>=end.d[DIR]; 
			i += dir)
		{
			s32 indT;
			if(DIR == 1)
				indT = i*w + (s32)start.x;	
			else 
				indT = (s32)start.y*w + i;

			if(indT<0 ||indT>=w*h)
				break;
			data[indT] = color;
		}
	}
	else 
	for(f64 t = 0; t<=1.f;)
	{
		V2 currentPoint = (1.-t)*start + t*end;
		s32 indT = (s32)(currentPoint.y)*w + (s32)currentPoint.x;
		if(indT<0 ||indT>=w*h)
			break;
		data[indT] = color;
		t+=addition;
	}
}
void DrawTriangle(V2 A,V2 B,V2 C,s32 w ,s32 h,u8*&data,RGB color)
{
	CreatePallateWithALineOnIt(A,B,w,h,data,color);
	CreatePallateWithALineOnIt(B,C,w,h,data,color);
	CreatePallateWithALineOnIt(C,A,w,h,data,color);
}

void DrawAABB(BB bb,s32 w ,s32 h,u8*&data,RGB color)
{
	V2 A = {bb.min[0],bb.min[1]};
	V2 B = {bb.max[0],bb.min[1]};
	V2 C = {bb.max[0],bb.max[1]};
	V2 D = {bb.min[0],bb.max[1]};
	CreatePallateWithALineOnIt(A,B,w,h,data,color);
	CreatePallateWithALineOnIt(B,C,w,h,data,color);
	CreatePallateWithALineOnIt(C,D,w,h,data,color);
	CreatePallateWithALineOnIt(D,A,w,h,data,color);
}
void FillAABB(BB bb,s32 w ,s32 h,u8*&data,RGB color)
{	
	V2 A = { bb.min[0] , bb.min[1] };
	V2 B = { bb.max[0] , bb.max[1] };
	if(B.x - A.x > B.y - A.y)
	{
		//this is longer in the x direction -
		B.y = bb.min[1];
		while(B.y <= bb.max[1])
		{
			CreatePallateWithALineOnIt(A,B,w,h,data,color);	
			B.y++;
			A.y++;
		}
	}
	else 
	{
		//this is longer in the y direction |
		B.x = bb.min[0];
		while(A.x <= bb.max[0])
		{
			CreatePallateWithALineOnIt(A,B,w,h,data,color);	
			A.x++;
			B.x++;
		}
	}
}

void DrawSirpensky(V2 A,V2 B,V2 C ,s32 n,s32 &w,s32 &h,u8*&data,RGB &color)
{
	if(n>=30)
		return;

	DrawTriangle(A,B,C,w,h,data,color);

	V2 A_0 = (1.f/2.f)*(A+B);
	V2 B_0 = (1.f/2.f)*(B+C);
	V2 C_0 = (1.f/2.f)*(C+A);	

	V2 A_1 = C_0;
	V2 B_1 = A;
	V2 C_1 = A_0;
	
	DrawSirpensky(A_1,B_1,C_1,n+1,w,h,data,color);

	V2 A_2 = A_0;
	V2 B_2 = B;
	V2 C_2 = B_0;
	
	DrawSirpensky(A_2,B_2,C_2,n+1,w,h,data,color);
	
	V2 A_3 = C_0;
	V2 B_3 = B_0;
	V2 C_3 = C;
	DrawSirpensky(A_3,B_3,C_3,n+1,w,h,data,color);
}

V2 perp(V2 a)
{
	return V(-a.y,a.x);
}

void DrawSirpensky(V2 A,V2 B,s32 n,s32 &w,s32 &h,u8*&data,RGB &color)
{
	if(n>=10)
		return;
	CreatePallateWithALineOnIt(A,B,w,h,data,color);
	
	V2 BMA = B-A;// aka a minus b
	f32 tHeight = sqrt(dot(BMA,BMA));

	V2 perpToBMA = perp(BMA);
	perpToBMA*=1.f/3.f;
	V2 middleP = (1.f/2.f)*(A+B);
	V2 top = middleP+perpToBMA;
	V2 left= A+(1.f/3.f)*(BMA);
	V2 right =A+(2.f/3.f)*(BMA);
	DrawTriangle(left,top,right,w,h,data,color);		 
	DrawSirpensky(top,right,n+1,w,h,data,color);
	DrawSirpensky(left,top,n+1,w,h,data,color);
	DrawSirpensky(A,left,n+1,w,h,data,color);
	DrawSirpensky(right,B,n+1,w,h,data,color);
}
//V2 rotate(f32 theta,V2 p)
//{
//	f32 sin_t = sin(theta);
//	f32 cos_t=  cos(theta);
//	return V(sin_t*p.x +
//}
//void DrawTree(V2 A,V2 B,s32 n,s32 &w,s32 &h,u8*&data,RGBA &color)
//{
//	if(n>=5)
//		return;
//	CreatePallateWithALineOnIt(A,B,w,h,data,color);	
//
//	V2 FP = A + (1.f/3.f)*(B-A);
//	V2 SP = B;
//
//	V2 left = 1/3*(B-A)
//}
void DrawMendelbrotSet(Z z,f32 r,s32 iter,s32 &w,s32 &h,u8*&data,RGB &color)
{
	FOR(y,0,h)
	{
		FOR(x,0,w)
		{
			Z c = {(f64)(x-h/2)/(h/2),(f64)(y-w/2)/(w/2)};
			Z Z_2 = c;
			FOR(k,0,iter)
			{
				f32 l = len(Z_2);
				if(l>r)
				{
					f32 dim = (f32 (k)/(iter));
					RGB base_color = {(1-dim)*255,(1-dim)*255,(1-dim)*255};
					RGB color_lr = {base_color.r+color.r*dim,base_color.g+color.g*dim,base_color.b+color.b*dim};
					//RGBA color_lr = color;
					CreatePallateWithALineOnIt(V(x,y),V(x,y),w,h,data,color_lr);
					break;
				}
				else if (k==(iter-1))
				{
					CreatePallateWithALineOnIt(V(x,y),V(x,y),w,h,data,color);	
				}
				else 
				{
					Z_2 =Z_2*Z_2 + c;
				}
					
			}

		}
	}
}
line_direction * CreateLineDirection(float * lengths,directions *dir,s32 count,orientations initialOr,V2 initialPos,f32 metersPerPixel)
{
	line_direction* ret;
	ret = (line_direction*)calloc(count,sizeof(line_direction));

	ret[0].position = initialPos;
	if(initialOr == HORIZONTAL)
		ret[0].direction = metersPerPixel*lengths[0]*V(-1,0);
	else 
		ret[0].direction = metersPerPixel*lengths[0]*V(0,1);

	for(s32 i = 1; i < count; i++ )
	{
		ret[i].position = ret[i-1].position + ret[i-1].direction;
		V2 previousDirectionNorm = 1.f/sqrt(dot(ret[i-1].direction,ret[i-1].direction)) * ret[i-1].direction;
		if(dir[i] == LEFT)
		{
				ret[i].direction.x = -previousDirectionNorm.y;
				ret[i].direction.y = previousDirectionNorm.x;
		}
		else
		{
				ret[i].direction.x = previousDirectionNorm.y;
				ret[i].direction.y = -previousDirectionNorm.x;
		}
		ret[i].direction*= lengths[i]*metersPerPixel;
	}
	return ret;
}

void CreatePerpPolygon(line_direction * ld,s32 count,u8* &toFill,s32 w,s32 h)
{
	RGB color = {255,0,255};
	for(s32 i = 0 ; i < count ; i++)
	{
		CreatePallateWithALineOnIt(ld[i].position,ld[i].position +ld[i].direction,w,h,toFill,color);
	}
}
void WriteDigit(s32 * minXY,s32 *maxXY,u8* img,s32 w ,s32 h,V2 startPos,f32 scaleFactor,image bmpFont)
{
		for(s32 ybmp = minXY[1]; ybmp <=maxXY[1] ; ybmp++)
		{
			for(s32 xbmp = minXY[0]; xbmp <=maxXY[0] ; xbmp++)
			{
				 s32 yimg =(startPos.y + (ybmp-minXY[1]) * scaleFactor),
					 ximg = startPos.x + (xbmp-minXY[0]) * scaleFactor;
					
				 if(yimg >= 0 && yimg < h && ximg > 0 && ximg < w)
				 {
					 RGB pixel = bmpFont.rgbdata[ybmp * bmpFont.w + xbmp];
					 if(pixel.r != 255 || pixel.g!=255 || pixel.b !=255)
						((RGB*)img)[yimg * w + ximg] = pixel;
				 }
			}
		}
}
void WriteNumber(char * str,s32 strLen,u8* img,s32 w,s32 h,V2 startPos,f32 scaleFactor)
{
	image bmpNumericFont;
	GetImage("BmpFont/numbersHQ.bmp",bmpNumericFont);
	{
			image t = Conv(bmpNumericFont,kernBlur[1]);
			free(bmpNumericFont.udata);
			bmpNumericFont = t;
	}
	if(!bmpNumericFont.udata) return;
	#define initMinMax(mnx,mny,mxx,mxy) minXY[0] = mnx,minXY[1] = mny,maxXY[0]=mxx,maxXY[1] =mxy
	FOR(i,0,strLen)
	{
		s32 minXY[] = {0,0},
			maxXY[] = {0,0};
		switch(str[i])
		{
		case '0':
			initMinMax(1083,3,1186,196);			
			break;
		case '1':
			initMinMax(1,3,59,193);
			break;
		case '2':
			initMinMax(105,3,210,196);
			break;
		case '3':
			initMinMax(230,3,332,196);
			break;
		case '4':
			initMinMax(345,3,454,193);
			break;
		case '5':
			initMinMax(474,5,577,196);
			break;
		case '6':
			initMinMax(594,3,698,196);
			break;
		case '7':
			initMinMax(718,5,821,193);
			break;
		case '8':
			initMinMax(840,3,942,196);
			break;
		case '9':
			initMinMax(961,3,1064,196);
			break;
		}
		if(minXY[0]!=0 || minXY[1] !=0 || maxXY[0] !=0 || maxXY[1]!=0);
		{
			WriteDigit(minXY,maxXY,img,w,h,startPos,scaleFactor,bmpNumericFont);
			startPos.x+= scaleFactor * (maxXY[0] - minXY[0] + 1) + (scaleFactor/4.7)*(maxXY[0] - minXY[0] + 1);
		}
			
	}
}