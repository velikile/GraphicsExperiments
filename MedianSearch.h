#include <stdio.h>
#include <math.h>
#include "TypeDefs.h"

unsigned long Arand();
void swap(float * arr,int i,int j)
{
	float t = arr[i];
	arr[i] = arr[j] ;
	arr[j] = t;
}
int partition(float * arr, int start,int end,float seed)
{	
	int i = start,j=end;
	while(i<j)
	{
		if(arr[i]>seed && arr[j]<seed)
		{
			swap(arr,i,j);
			j--;
			i++;
		}
		else if(arr[j] >=seed)
			j--;
		else if(arr[i] <=seed)
			i++;
	}
	return (i+j)/2;
}

int findMedian(float * arr, int start,int end,int medIndex)
{
	if(start == end)
		return -1;
	int seedIndex = Arand()% (end+1) + start;
	seedIndex = seedIndex > end ? end:seedIndex;
	int seperationPoint = partition(arr,start,end,arr[seedIndex]);
	if(seperationPoint == medIndex)
		return arr[medIndex];
	else 
	{
		if(seperationPoint>medIndex)
			return findMedian(arr,start,seperationPoint,medIndex);
		else 
			return findMedian(arr,seperationPoint+1,end,medIndex);
		
	}
}