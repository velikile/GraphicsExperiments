#ifndef NEURAL_NET
#define NEURAL_NET
#include <math.h>
#include "TypeDefs.h"
#include <Windows.h>



struct h_layer
{
	s32 nCount;
	//weights and biases are used for the weights going into the layer from the previous layer 
	f32 * weights;
	f32 * biases;
	f32 * input;
	f32 * output;
	f32 * error;
};

struct neural_net
{
	s32 inputCount;
	s32 hiddenLayersCount;
	s32 outputCount;
	f32 learningRate;
	f32 * input;
	h_layer* hiddenLayers;
	h_layer output;
};

inline f32 Sigmoid(f32 val)
{
	return 1.f/(1.f + 1.f/pow(2.7182818284f,val));
}

inline f32 DSigmoid_d(f32 val)
{
	return (1.f - val) * val;
}
inline f32 ReLU(f32 val)
{
	return val < 0 ? 0 : val;
}
inline f32 DReLU_d(f32 val)
{
	return val < 0? 0: 1;
}

f32 Activate(f32 val)
{
	return ReLU(val);
}

f32 DActivate_d(f32 val)
{
	return DReLU_d(val);
}

f64 * weights = new f64[4];
f64 * biases =  new f64[4];

neural_net CreateNetwork(s32 inputCount,s32 hiddenLayersCount,s32 * countPerHiddenLayer,s32 outputCount,f32 learningRate=0.01f)
{
	neural_net ret = {inputCount,hiddenLayersCount,outputCount,learningRate};
	ret.hiddenLayers = new h_layer[hiddenLayersCount];
	ret.input = new f32[inputCount];
	ret.outputCount = outputCount;
	s32 prevCount = inputCount; // number of connections to the previous layer
	h_layer *currentLayer = 0;
	s32 countPerLayer = 0;
	for(s32 i  =0;i<=hiddenLayersCount;i++)
	{
		if(i == hiddenLayersCount)
		{
			currentLayer = &ret.output;
			countPerLayer = outputCount;
		}
		else 
		{
			currentLayer = &ret.hiddenLayers[i];
			countPerLayer = countPerHiddenLayer[i];
		}
		
		currentLayer->weights = new  f32  [countPerLayer * prevCount];
		currentLayer->biases  = new  f32  [countPerLayer * prevCount];
		currentLayer->input   = (f32*)calloc(countPerLayer,sizeof(f32));
		currentLayer->output  = (f32*)calloc(countPerLayer,sizeof(f32));
		currentLayer->error  =  (f32*)calloc(countPerLayer,sizeof(f32));
		currentLayer->nCount = countPerLayer;
		
		assert(currentLayer->weights && currentLayer->biases);
		
		for(s32 j= 0;j<countPerLayer*prevCount;j++)
		{	
			currentLayer->weights[j] = sin((f32)rand());
			currentLayer->biases[j] = sin((f32)rand());
		}
		prevCount = countPerLayer;
	}
	return ret;
}
void FillInput(neural_net &net,f32 *input,s32 inputLength)
{
	assert(net.inputCount<=inputLength);
	for(s32 i = 0; i < inputLength;i++)
	{
		net.input[i] = input[i];
	}
}
void ProduceOutput(neural_net net)
{
	s32 prevLayerCount = net.inputCount;
	h_layer * currentLayer = 0;
	for(s32 j=0;j<=net.hiddenLayersCount;j++)
	{
		if(net.hiddenLayersCount == j)
			currentLayer = &net.output;
		else currentLayer = &net.hiddenLayers[j];
		for(s32 s = 0; s<currentLayer->nCount; s++)
		{	
			currentLayer->input[s] = 0;
			currentLayer->output[s] = 0;
			currentLayer->error[s] = 0;
			for(s32 k=0;k<prevLayerCount;k++)
			{
				f32 currentInput;
				if(j==0)
					currentInput = net.input[k];
				else
					currentInput = net.hiddenLayers[j-1].output[k];

				s32 ind = s*prevLayerCount + k;
				currentLayer->input[s] += currentLayer->weights[ind] * currentInput  +
											    currentLayer->biases[ind];
			}
			
			currentLayer->output[s] = Activate(currentLayer->input[s]);
		}
		prevLayerCount = currentLayer->nCount;
	}
}
void BackPropFixValue(neural_net & net,f32 *target,f32 learningRate)
{
	// this process goes from output back to input reversed from ProduceOutput way
	ProduceOutput(net);
	for(s32 i = 0; i < net.outputCount ; i++)
	{
		net.output.error[i] = (net.output.output[i]- target[i]);
	}
	s32 prevLayerNCount = net.outputCount;
	h_layer * prevLayer = &net.output;
	for(s32 h = net.hiddenLayersCount -1;h>=0;h--)
	{
		for(s32 i = 0 ;  i < prevLayerNCount ; i++)
		{
			for(s32 k = 0; k<net.hiddenLayers[h].nCount; k++)
			{
				net.hiddenLayers[h].error[k] += (prevLayer->error[i]*
												DActivate_d(prevLayer->input[i])*
												prevLayer->weights[k*prevLayerNCount + i])/net.hiddenLayers[h].nCount;
			}
		}
		for(s32 j = 0 ; j <prevLayerNCount;j++)
		{
			for(s32 k = 0 ; k < net.hiddenLayers[h].nCount ;k++)
			{
				prevLayer->weights[j*net.hiddenLayers[h].nCount + k] -=	 learningRate * (prevLayer->error[j] *
																		 net.hiddenLayers[h].output[k])/prevLayerNCount;
				prevLayer->biases[j*net.hiddenLayers[h].nCount + k]  -=  learningRate * (prevLayer->error[j])/prevLayerNCount;
			}
		}
		prevLayer = &net.hiddenLayers[h];
		prevLayerNCount = net.hiddenLayers[h].nCount;
	}

	for(s32 j = 0 ; j < prevLayerNCount; j++)
	{	
		for(s32 k = 0; k <net.inputCount; k++)
		{
			prevLayer->weights[j*net.inputCount+ k] -= learningRate * ((prevLayer->error[j] * net.input[k])/prevLayerNCount);
			prevLayer->biases[j*net.inputCount+ k]  -= learningRate * prevLayer->error[j]/prevLayerNCount;
		}
	}
}
#endif
