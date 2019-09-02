#ifndef PERF_MON
#define PERF_MON

#include <windows.h>

double PCFreq = 0.0;

bool StartCounter(__int64 &CounterStart)
{
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
		return false;
	if(PCFreq == 0.0)
		PCFreq = double(li.QuadPart)/1000.0;
	
    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
	return true;
}

double GetCounter(__int64 CounterStart)
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-CounterStart)/PCFreq;
}
#define StartCounterA(CounterName) __int64 CounterName; StartCounter(CounterName)

#endif