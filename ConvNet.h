#ifndef CONV_NET
#define CONV_NET
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "TypeDefs.h"

struct image
{
	s32 w;
	s32 h;
	union
	{
		u8 * udata;
		RGB * rgbdata;
		RGBA * rgbadata;
		f32 * fdata;
	};
};

struct imageData
{
	unsigned char header[54]; // Each BMP file begins by a 54-bytes header
	unsigned int dataPos;     // Position in the file where the actual data begins
	unsigned int width, height;
	unsigned int imageSize;   // = width*height*3
	unsigned char * data;
	unsigned int rowSize;
};
void GetImage(s8 * imagepath,image &img)
{
	img.udata = 0;
	imageData res;
	FILE * file = fopen(imagepath,"rb");
	if (!file)
	{
		printf("Image could not be opened\n");
	 	return;
	}
	if ( fread(res.header, 1, 54, file)!=54 )
	{
	    printf("Not a correct BMP file\n");
	    return;
	}
	if ( res.header[0]!='B' || res.header[1]!='M' )
	{
		printf("Not a correct BMP file\n");
		return;
	}
	res.dataPos    = *(int*)&(res.header[0x0A]);
	res.imageSize  = *(int*)&(res.header[0x22]);
	img.w = *(int*)&(res.header[0x12]);
	img.h = *(int*)&(res.header[0x16]);
	s32 padding = img.w%4;//(4-(img.w*3)% 4)%4;
	if (res.imageSize==0)
	    res.imageSize=img.w * img.h * 3; // 3 : one byte for each Red, Green and Blue component

	img.udata = new unsigned char[res.imageSize];
	u8 * currentDataPtr = img.udata;
	while(true)
	{
		if (fread(currentDataPtr,3 ,img.w,file)!= img.w)
			break;
		fseek(file,padding,SEEK_CUR);
		currentDataPtr += img.w*3;
	}

	for(s32 i = 0; i<img.w*img.h*3;i+=3)
	{
			///u8 temp = img.udata[i];
			///img.udata[i] = img.udata[i +2];
			///img.udata[i +2]= temp;
	}
	fclose(file);
}

void freeImage(imageData*d, bool freeData)
{
	if( d && freeData)
	{
		if(d->data)
		{
			free(d->data);
		}
		free(d);
	}
}
RGB RGBmax(RGB a, RGB b)
{
	RGB ret = {a.r > b.r ?a.r:b.r,
			   a.g > b.g ?a.g:b.g, 
			   a.b > b.b ?a.b:b.b};
	return ret;
}
image CreateImageRGB(s32 w, s32 h)
{
	image res={w,h};
	res.rgbdata = (RGB*)calloc(w*h,sizeof(RGB));
	assert(res.rgbdata);
	return res;
}
image CreateImageRGBA(s32 w, s32 h)
{
	image res={w,h};
	res.rgbdata = (RGB*)calloc(w*h,sizeof(RGB));
	assert(res.rgbdata);
	return res;
}


RGB f3f32ToRGB(f32 * resf)
{
	RGB res = {max(0,min(resf[0],255)),
				max(0,min(resf[1],255)),
				max(0,min(resf[2],255))
				//max(0,min(resf[3],255))
	};
	return res;
}
image ConvR(image &img,image & kern)
{
	image ret = CreateImageRGB(img.w-(kern.w-1),img.h-(kern.h-1));
	s32 counter = 0;
	for(s32 j = 0 ; j < ret.h  ; j++)
	{
		for(s32 i = 0 ; i < ret.w; i++)
		{
			f32 resf[3] ={0};
			for(s32 m = 0 ; m < kern.h ; m++)
			{
				for(s32 k = 0; k < kern.w ; k++)
				{
					s32 imageIndex = (j+m)*img.w + i + k;
					s32 kernIndex =  m * kern.w + k;
					resf[0] += (f32)img.rgbdata[imageIndex].r * kern.fdata[kernIndex];
					resf[1] += (f32)img.rgbdata[imageIndex].g * kern.fdata[kernIndex];
					resf[2] += (f32)img.rgbdata[imageIndex].b * kern.fdata[kernIndex];
				}
			}
				RGB res = f3f32ToRGB(resf);
				ret.rgbdata[counter++] = res;
		}
	}
	return ret;
}
image Conv(image &img,image & kern)
{
	image ret = CreateImageRGB(img.w,img.h);
		
	s32 offsetW = kern.w/2;
	s32 offsetH = kern.h/2;
	s32 limH = img.h - offsetH;
	s32 limW = img.w - offsetW;
	for(s32 j = -kern.h/2 ; j < limH  ; j++)
	{
		for(s32 i = -kern.w/2 ; i < limW ; i++)
		{
			f32 resf[3] ={0};

			s32 sH = j<0 ? -j : 0;
			s32 sW = i<0 ? -i : 0;
			s32 fH = j+kern.h>=img.h ? kern.h - (img.h - j) : kern.h;
			s32 fW = i+kern.w>=img.w ? kern.w - (img.w - i) : kern.w;
			
			for(s32 m = sH ; m < fH ; m++)
			{
				for(s32 k = sW; k < fW ; k++)
				{	
					s32 imageIndex = (j+m)*img.w + i + k;
					s32 kernIndex =  m * kern.w + k;
					resf[0] += (f32)img.rgbdata[imageIndex].r * kern.fdata[kernIndex];
					resf[1] += (f32)img.rgbdata[imageIndex].g * kern.fdata[kernIndex];
					resf[2] += (f32)img.rgbdata[imageIndex].b * kern.fdata[kernIndex];
				}
			 }
				RGB res = f3f32ToRGB(resf);
				ret.rgbdata[(j+offsetH)*img.w + (i + offsetW)] = res;
		}
	}
	return ret;
}
enum layer_type
{
	INPUT_OUTPUT,
	HIDDEN
};
struct forward_link_weight
{
	f32 * weight;
	s32 * nextLayerIndices;
	s32 connectionsCount;
};
struct conv_layer;
struct conv_neuron
{
	f32 * weightsToPreviousLayer;
	s32 weightsToPreviousLayerCount;
	s32 * indicesInPreviousLayer;
	f32 preActivationR,postActivationR;
	f32 preActivationG,postActivationG;
	f32 preActivationB,postActivationB;
	conv_layer * prev; 
};
struct conv_layer
{
	s32 w;
	s32 h;
	s32 d;
	s32 neuronCount;
	conv_neuron * neurons;
	layer_type lType;
};
struct conv_net
{
	s32 layerCount;
	conv_layer * layers;
};
conv_layer ConvertImageToConvInput(image & img)
{
	conv_layer ret={0};
	ret.w = img.w;
	ret.h = img.h;
	ret.d = 1;
	ret.neuronCount = ret.h * ret.w;
	ret.lType = INPUT_OUTPUT;
	ret.neurons = (conv_neuron*)calloc(ret.neuronCount,sizeof(conv_neuron));
	assert(ret.neurons);
	for(s32 i=0 ; i<ret.neuronCount ; i++)
	{	
		ret.neurons[i].postActivationR = (f32)img.rgbdata[i].r/255.f;
		ret.neurons[i].postActivationG = (f32)img.rgbdata[i].g/255.f;
		ret.neurons[i].postActivationB = (f32)img.rgbdata[i].b/255.f;
	}
	return ret;
}
image ConvertConvLayerToImage(conv_layer layer,s32 filterIndex)
{
	image ret = CreateImageRGB(layer.w,layer.h);
	s32 sIndex = layer.w *layer.h *filterIndex;
	s32 fIndex = layer.w * layer.h*(filterIndex +1);
	for(s32 i = sIndex,im=0; i<fIndex ;i++,im++)
	{
		f32 t[] = { layer.neurons[i].postActivationR * 255,
					layer.neurons[i].postActivationG * 255,
					layer.neurons[i].postActivationB * 255};
		ret.rgbdata[im] = f3f32ToRGB(t);
	}
	return ret;
}
conv_layer *CreateConvLayer(conv_layer *cLayer,image* kern,V2 stride ,s32 depth)
{
	s32 previousLayerCount = cLayer->w * cLayer->h * cLayer->d;

	conv_layer * ret = (conv_layer *) calloc(1,sizeof(conv_layer));
	
	ret->w = cLayer->w;
	ret->h = cLayer->h;
	ret->d = depth;
	ret->lType = HIDDEN;
	s32 limH = ret->h;
	s32 limW = ret->w;
	s32 jc=0,ic=0,neuronCount = 0; // running counters
	ret->neuronCount = ret->w * ret->h * depth;
	ret->neurons =  (conv_neuron *)calloc(ret->neuronCount,sizeof(conv_neuron));
	
	for(s32 d = 0 ; d < depth ; d++) // currentLayerDepth
	{	s32 offsetW = kern[d].w/2;
		s32 offsetH = kern[d].h/2;
		s32 weightsPerNeuron = kern[d].w*kern[d].h;
		jc=0;
		for(s32 dp = 0 ;dp< cLayer->d;dp++) // previousLayerDepth
		{
			for(s32 j = 0;j < limH  ; j+=stride.yi,jc++)
			{
				ic = 0;
				for(s32 i = 0; i < limW ; i+=stride.xi,ic++)
				{
					s32 sH = j < offsetH ? offsetH - j  : 0;
					s32 sW = i < offsetW ? offsetW - i : 0;
					s32 fH =kern[d].h - max(j - (ret->h - kern[d].h + offsetH),0);
					s32 fW =kern[d].w - max(i - (ret->w - kern[d].w + offsetW),0);
					ret->neurons[neuronCount].weightsToPreviousLayer = (f32*) calloc(weightsPerNeuron,sizeof(f32));
					ret->neurons[neuronCount].indicesInPreviousLayer = (s32*) calloc(weightsPerNeuron,sizeof(f32));
					ret->neurons[neuronCount].prev = cLayer;
					s32 kernIndex = 0;
					for(s32 m = sH ; m < fH ; m++)
					{
						for(s32 k = sW; k < fW ; k++)
						{
							s32 prevIndex = dp*cLayer->w*cLayer->h + (j-offsetH+m) * ret->w + i - offsetW + k;
							ret->neurons[neuronCount].weightsToPreviousLayer[kernIndex] = kern[d].fdata[kernIndex];
							ret->neurons[neuronCount].indicesInPreviousLayer[kernIndex] = prevIndex ;
							kernIndex++;
						}
					 }
					 ret->neurons[neuronCount].weightsToPreviousLayerCount = kernIndex;
					 if(kernIndex)
					 {
					 	ret->neurons[neuronCount].weightsToPreviousLayer = (f32*)realloc(ret->neurons[neuronCount].weightsToPreviousLayer,kernIndex*sizeof(f32));
					 	neuronCount++;
					 }
					 else
					 {
					 	s32 x= 0;
					 }
				}
			}
		}
	}
	if(ic != ret->w || jc !=ret->h)
	{
		ret->w = ic;
		ret->h = jc;
		ret->neuronCount = neuronCount;
		ret->neurons = (conv_neuron*)realloc(ret->neurons,ret->neuronCount*sizeof(conv_neuron));
		assert(ret->neurons);
	}
	return ret;
	
}
void AddLayerToNetwork(conv_net&network, conv_layer layer)
{
	network.layerCount++;
	network.layers = (conv_layer*)realloc(network.layers,network.layerCount*sizeof(conv_layer));
	network.layers[network.layerCount-1] = layer;
}
void ComputeConvOutput(conv_net &network)
{
	// for this method of compting the output the output will be at the network.layerCount-1 position
	for(s32 l = 1;l < network.layerCount;l++) // skip the input layer
	{
		for(s32 n=0 ; n < network.layers[l].neuronCount; n++)
		{
			s32 weightCount = network.layers[l].neurons[n].weightsToPreviousLayerCount;
			for(s32 w = 0 ; w< weightCount ;w++)
			{
				network.layers[l].neurons[n].preActivationR += network.layers[l-1].neurons[network.layers[l].neurons[n].indicesInPreviousLayer[w]].postActivationR * network.layers[l].neurons[n].weightsToPreviousLayer[w];
				network.layers[l].neurons[n].preActivationG += network.layers[l-1].neurons[network.layers[l].neurons[n].indicesInPreviousLayer[w]].postActivationG * network.layers[l].neurons[n].weightsToPreviousLayer[w];
				network.layers[l].neurons[n].preActivationB += network.layers[l-1].neurons[network.layers[l].neurons[n].indicesInPreviousLayer[w]].postActivationB * network.layers[l].neurons[n].weightsToPreviousLayer[w];
			}
			network.layers[l].neurons[n].postActivationR = Activate(network.layers[l].neurons[n].preActivationR);
			network.layers[l].neurons[n].postActivationG = Activate(network.layers[l].neurons[n].preActivationG);
			network.layers[l].neurons[n].postActivationB = Activate(network.layers[l].neurons[n].preActivationB);
		}
	}
}


void Invert(image&img)
{
	for(s32 i =0;i<img.w*img.h*3;i++)
	{
		img.udata[i]  = 255 - img.udata[i];
	}
}
enum image_ops{ADD,SUB,AVG,MAX};
image Combine(image &a ,image &b,image_ops op)
{
	s32 minW = min(a.w,b.w);
	s32 minH = min(a.h,b.h);
	image ret = CreateImageRGB(minW,minH);
	for(s32 x = 0 ; x < minW; x++)
		for(s32 y = 0 ; y< minH ;y++)
		{
			s32 currentPixel=x*minH + y;
			switch(op)
			{
				case ADD:
					ret.rgbdata[currentPixel] = a.rgbdata[currentPixel] + b.rgbdata[currentPixel];
					break;
				case SUB:
					ret.rgbdata[currentPixel] = a.rgbdata[currentPixel] - b.rgbdata[currentPixel];
					break;
				case AVG:
					ret.rgbdata[currentPixel] = (a.rgbdata[currentPixel] + b.rgbdata[currentPixel])/2;
					break;
				case MAX:
					ret.rgbdata[currentPixel] = RGBmax(a.rgbdata[currentPixel],b.rgbdata[currentPixel]);
					break;
			}
		}
		return ret;
}
void Treshold(image& img,u8 limit)
{
	s32 pixelCount = img.w*img.h*3;
	for(s32 i =0;i<pixelCount;i+=3)
	{
		s32 pixelAvg = (img.udata[i] + img.udata[i+1] + img.udata[i+2])/3; // bw pixel value sorta ...
		if(pixelAvg > limit )
		{
			img.udata[i] = 255;
			img.udata[i+1] = 255;
			img.udata[i+2] = 255;
		}
		else
		{ 
			img.udata[i] = 0;
			img.udata[i+1] = 0;
			img.udata[i+2] = 0;
		}
	}
}
image DownSampleRGB(image & img,s32 w,s32 h)
{
	image ret= CreateImageRGB(w,h);
	for(s32 i = 0; i < h ;i++)
	{
		for(s32 j =0; j<w; j++)
		{
				ret.rgbdata[i *w + j] = img.rgbdata[(img.w*j)/w + ((img.h*i)/h)*img.w];
		}
	}
	return ret;
}
image MaxPool(image & img, s32 windowSize,s32 stride)
{		
	image ret= CreateImageRGB(img.w/stride,img.h/stride);
	s32 w = ret.w;
	s32 h = ret.h;
	for(s32 i = 0,ni=0; ni < h ;i+=stride,ni++)
	{
		for(s32 j =0,nj=0; nj< w; j+=stride,nj++)
		{
			RGB pixel = {0,0,0};
			for(s32 k = 0; k <windowSize; k++)
			{
				for(s32 l = 0;l <windowSize; l++)
				{
					s32 hIndex = i+k;
					s32 wIndex = j+l;
					if(hIndex <= img.h -1 && wIndex <= img.w -1 )
						pixel = RGBmax(pixel,img.rgbdata[hIndex*img.w + wIndex]);
				}
			}
			ret.rgbdata[w*ni + nj] = pixel;
		}
	}
	return ret;
}
//#define FILTERS_COUNT 8
//
//void ForwardConv(conv_net net,image img)
//{
//	image * images = ProduceFiltersVersions(img);
//	image pooledImages[FILTERS_COUNT];
//	for(s32 i =0 ; i < FILTERS_COUNT ; i++)
//	{
//		pooledImages[i] = MaxPool(images[i],3,3);
//	}
//	ConvertToFlatInputForNetwork(pooledImages);
//
//}

#endif