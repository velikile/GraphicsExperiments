#include <malloc.h>
#include "TypeDefs.h"
#include "PerfMonitor.h"

#define MAX_DEPTH 11
#define DEVIDE_QUADRANT_TRASHOLD 15
#define MAX_DATA_AMOUNT DEVIDE_QUADRANT_TRASHOLD

struct BB
{
	float min[2];
	float max[2];
};
enum QUADRENT{LEFT_TOP,RIGHT_TOP,LEFT_BOTTOM,RIGHT_BOTTOM,QUADSIZE};
struct QuadNode
{
	bool hasData;
	float data[MAX_DATA_AMOUNT][2];
	int currentTop;
	QuadNode * parent;
	QuadNode * Nodes[4];
	float dataSum[2];
	bool bbInitialized;
	BB bb;
};

void insertIntoTree(QuadNode * t, float item[2], int depth);

bool ContainedInBB(BB bb,float item[2])
{
	return bw(item[0],bb.min[0],bb.max[0]) && bw(item[1],bb.min[1],bb.max[1]);
}
BB getChildNodeBB(BB parent,int index)
{
	BB result={0};
	switch(index)
	{
		case LEFT_TOP:
			result.min[0] = parent.min[0];
			result.min[1] = parent.min[1];
			result.max[0] = (parent.min[0] + parent.max[0])/2.f;
			result.max[1] = (parent.min[1] + parent.max[1])/2.f;
			break;
		case RIGHT_TOP:
			result.min[0] = (parent.min[0] + parent.max[0])/2.f;
			result.min[1] = parent.min[1];
			result.max[0] = parent.max[0];
			result.max[1] = (parent.min[1] + parent.max[1])/2.f;
			break;
		case LEFT_BOTTOM:
			result.min[0] = parent.min[0];
			result.min[1] = (parent.min[1] + parent.max[1])/2.f;;
			result.max[0] = (parent.min[0] + parent.max[0])/2.f;
			result.max[1] = parent.max[1];
			break;
		case RIGHT_BOTTOM:
			result.min[0] = (parent.min[0] + parent.max[0])/2.f;
			result.min[1] = (parent.min[1] + parent.max[1])/2.f;
			result.max[0] = parent.max[0];
			result.max[1] = parent.max[1];
			break;
	};
	return result;
}
QuadNode * pickNode(QuadNode *t,float item[2])
{	
	if(t && ContainedInBB(t->bb,item))
	{
		for(int i = LEFT_TOP ; i< QUADSIZE;i++)
		{	BB tempBB = getChildNodeBB(t->bb,i);
			if(ContainedInBB(tempBB,item))
			{
				if(t->Nodes[i])
					return t->Nodes[i];
				else 
				{
					t->Nodes[i] = (QuadNode*)malloc(sizeof(QuadNode));
					memset(t->Nodes[i],0,sizeof(QuadNode));
					t->Nodes[i]->bb = tempBB;
					return t->Nodes[i];
				}
			}
		}
	}
	return 0;
}
u8 DataBlock[1<<30];
u32 CurrentDataPoint = 0;
inline void * MyAlloc(s32 size)
{
	if(CurrentDataPoint > ((1<<30) -1))
		return 0;
	void *ret = DataBlock + CurrentDataPoint;
	CurrentDataPoint+=size;
	return ret;
}
inline void MyFreeIm(void * ptr)
{
	if((u32)ptr - (u32)DataBlock >= 0)
	{
		CurrentDataPoint = (u32)ptr - (u32)DataBlock;
	}
}

void CreateDividedNode(QuadNode *t)
{	
	if(t->Nodes[0])
		return;
	for(int i = LEFT_TOP ; i< QUADSIZE;i++)
	{
		if(!t->Nodes[i])
		{
				//t->Nodes[i] = (QuadNode*)malloc(sizeof(QuadNode));
				t->Nodes[i] = (QuadNode*)MyAlloc(sizeof(QuadNode));
				memset(t->Nodes[i],0,sizeof(QuadNode));
				t->Nodes[i]->bb = getChildNodeBB(t->bb,i);
				for(int j = t->currentTop-1; j>=0 ; j--)
				{
					if(ContainedInBB(t->Nodes[i]->bb,t->data[j]))
					{
						//StartCounterA(insertCounter);
						insertIntoTree(t->Nodes[i],t->data[j],0);
						//printf("insertCounterTime: %.5f\n",GetCounter(insertCounter));
					}
				}
		}
	}
}
void insertIntoTree(QuadNode * t, float item[2], int depth = 0)
{
	while(t)
	{
		if (depth == MAX_DEPTH || t->currentTop < DEVIDE_QUADRANT_TRASHOLD)
		{
			if(t->currentTop >= MAX_DATA_AMOUNT)
				return;
			for(s32 i = 0; i < t->currentTop ; i++)
			{
				if(t->data[i][0] == item[0] && 
				   t->data[i][1] == item[1])
					return;
			}
			t->data[t->currentTop][0] = item[0];
			t->data[t->currentTop++][1] = item[1];
			t->dataSum[0] +=item[0];
			t->dataSum[1] +=item[1];
			t->hasData = true;
			break;
		}
		else
		{	
			CreateDividedNode(t);
			QuadNode * newT = pickNode(t,item);
			newT->parent = t;
			while(newT->Nodes[0])
			{
				newT = pickNode(newT,item);
				depth++;
			}
			t = newT;
		}
	}
}


void initQuadNode(QuadNode & node ,BB bb)
{
	node.bb = bb;
	node.bbInitialized = true;
	node.currentTop = 0;
	node.hasData = false;
	node.parent = 0;
	node.Nodes[0] = 0;
	node.Nodes[1] = 0;
	node.Nodes[2] = 0;
	node.Nodes[3] = 0;
}
void usageCode()
{
	QuadNode head={0};
	float toInsert[2] = {200,200};
	BB allSpace = {0,0,200,200};
	initQuadNode(head,allSpace);

}