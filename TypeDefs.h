#ifndef TYPE_DEF
#define TYPE_DEF

typedef unsigned char u8;
typedef char s8;
typedef bool b8;
typedef short s16;
typedef unsigned short u16;
typedef int s32;
typedef unsigned int u32;
typedef long long s64;
typedef  float f32 ;
typedef double f64 ;


union V2
{
	struct {f32 x,y;};
	struct {s32 xi,yi;};
	f32 d[2];
};
V2 V(f32 x,f32 y)
{
	V2 ret = {x,y};
	return ret;
}
#define bw(val,s,f)((val)>=(s) && (val)<=(f))

struct RGB
{
	u8 r,g,b;
};
struct RGBA
{
	u8 r,g,b,a;
};

RGB operator - (RGB colorA , RGB colorB)
{
	RGB ret = {max(0,colorA.r - colorB.r),
			   max(0,colorA.g - colorB.g),
			   max(0,colorA.b - colorB.b)};
	return ret;
}

RGB operator + (RGB colorA , RGB colorB)
{
	RGB ret = {min(255,colorA.r + colorB.r),
			   min(255,colorA.g + colorB.g),
			   min(255,colorA.b + colorB.b)};
	return ret;
}
RGB operator / (RGB colorA , s32 b)
{
	RGB ret = {colorA.r / b,
			   colorA.g / b,
			   colorA.b / b};
	return ret;
}



void operator += (RGBA &colorA,RGBA &colorB)
{
	colorA.a = colorA.a + colorB.a;
	colorA.r = colorA.r + colorB.r;
	colorA.g = colorA.g + colorB.g;
	colorA.b = colorA.b + colorB.b;
}

RGBA operator *(RGBA & colorA,RGBA &colorB)
{
	RGBA res={colorA.a * colorB.a,
			  colorA.r * colorB.r,
			  colorA.g * colorB.g,
			  colorA.b * colorB.b};
	return res;
}


#endif