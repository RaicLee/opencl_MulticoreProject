/*
2020/11~2020/12/7
Sejong University, Department of Computer Engineering 
16011105 Jeasung Lee / OPENCL CNN ACCELERATE PROJECT KERNEL
*/
//1 .Naive하게 개발한 코드
//여긴 사용안함
__kernel void fc_layer_multi(__global float *Input,
							 __global float *Weight,
							 __global float *Bias,
							 __global float *Output,
							 int N)
{
	int x =get_global_id(0);//0~M-1 output
	float sum = 0;
	int i;

	for(i = 0;i < N;i++)
		sum+= Input[i] * Weight[x * N + i];
	sum+= Bias[x];
	Output[x]=(sum>0)?sum:0;
}
__kernel void pooling2x2(__global float *Input,
						 __global float *Output)
{
	int d=get_global_id(0);//0~D-1
	int y=get_global_id(1);//0~N-1
	int x=get_global_id(2);//0~N-1
	int N=get_global_size(1);
	float max;
	int base = d * N * N;
	float4 result;
	result.x=Input[4*base+(y * 2 ) * 2 * N + x * 2 + 0];
	result.y=Input[4*base+(y * 2 ) * 2 * N + x * 2 + 1];
	result.z=Input[4*base+(y * 2 + 1) * 2 * N + x * 2 + 0];
	result.w=Input[4*base+(y * 2 + 1) * 2 * N + x * 2 + 1];
	max = (result.x>result.y)?result.x:result.y;
	max = (max>result.z)?max:result.z;
	max = (max>result.w)?max:result.w;
	Output[base+y*N +x]=max;
}
__kernel void conv(__global float *Input,
				   __global float *Output,
				   __global float *Weight,
				   __global float *Bias,
				   int N, int D1,int D2)
{
	int globalindex= get_global_id(0);
	int batchindex = get_global_id(1);
	int outputbase = batchindex * N * N * D2 + N * N * globalindex;
	int inputbase;
	int weightbase;
	float B=Bias[globalindex];
	int bi, i, j, k, l, x, y;
	float temp;
	float sum;
	float partialsum;

	for(bi=0;bi<D1;bi++)
	{
		inputbase =  batchindex*N*N*D1+N * N * bi;
		weightbase = 3 * 3 * (globalindex * D1 + bi);
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				sum = 0;
				partialsum=Output[outputbase + i * N + j];

				x=i-1;
				y=j-1;
				if (x >= 0 && y >= 0)
							sum += Input[inputbase + x * N + y] * Weight[weightbase];
				y=j;
				if (x >= 0)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 1];
				y=j+1;
				if (x >= 0 && y < N)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 2];

				x=i;
				y=j-1;
				if (y >= 0)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 3];
				y=j;
				sum += Input[inputbase + x * N + y] * Weight[weightbase+ 4];
				y=j+1;
				if (y < N)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 5];
				x=i+1;
				y=j-1;
				if (x < N && y >= 0)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 6];
				y=j;
				if (x < N)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 7];
				y=j+1;
				if (x < N && y < N)
							sum += Input[inputbase + x * N + y] * Weight[weightbase+ 8];
				
				Output[outputbase+ i * N + j] = partialsum+sum;
			}
		}
	}
	for(k=0;k<N*N;k++)
	{
		temp = Output[outputbase + k] + B;
		Output[outputbase+k]=ReLU(temp);
	}
}



//2. 배치용
//요거를 사용
__kernel void fc_layer_BATCH(__global float *Input,
							 __global float *Weight,
							 __global float *Bias,
							 __global float *Output,
							 int M, int N) {
    int globalindex = get_global_id(0);
	if(globalindex >= M) return;

    int batchindex  = get_global_id(1);
	__global float *localInput = Input + (batchindex * N); //InputSize
    __global float *localOutput = Output + (batchindex * M); //OutputSize
    float sum=0;

    for (int i = 0; i < N; i++)
        sum += localInput[i] * Weight[globalindex * N + i];

    sum += Bias[globalindex];
    localOutput[globalindex] = ReLU(sum);
}
__kernel void pooling2x2_BATCH(__global float *Input,
							   __global float *Output,
							   int D, int N) 
{
    int globalindex = get_global_id(0);
	if(globalindex >= (D * N * N)) return;

    int batchindex = get_global_id(1);
    __global float *input = Input + (batchindex * N * N * D * 4);
    __global float *output = Output + (batchindex * N * N * D);
    int base = globalindex / (N * N); //D 찾기
    int y = (globalindex / N) % N; //NXN안에서 y찾기
    int x = globalindex % N; //나머지에서 x찾기
	float4 values;
	float max;

	values.x=input[(base * N * N * 4) + ((y * 2) * 2 * N + x * 2 )];
	values.y=input[(base * N * N * 4) + ((y * 2) * 2 * N + x * 2 + 1)];
	values.z=input[(base * N * N * 4) + ((y * 2 + 1) * 2 * N + x * 2 )];
	values.w=input[(base * N * N * 4) + ((y * 2 + 1) * 2 * N + x * 2 + 1)];
	max = MAX(values.x,values.y);
	max = MAX(max,values.z);
	max = MAX(max,values.w);
    output[(base * N * N) + (y * N + x)] = max;
}
__kernel void conv_BATCH(__global float *Input,
						 __global float *Output,
						 __global float *Weight,
						 __global float *Bias,
						 int N, int D1,int D2)
{
	//1. N*N용
	int localindex= get_global_id(0); //N*N
	if(localindex >= N * N) return;
	//2. D용
	int dimensionindex = get_global_id(1);//D2
	if(dimensionindex >= D2) return;
	//3. BATCH용
	int batchindex = get_global_id(2);// 배치용
	int inputbase;
	int weightbase;
	float sum=0;
	float B=Bias[dimensionindex];
	int hy,hx,bi,x,y,k,l;
	hy = localindex/N;
	hx = localindex%N;

	__global float *output=Output+ batchindex * D2 * N * N + N * N * dimensionindex;
	
	//각각의 Local내에서 
	for(bi=0;bi<D1;bi++)//input에 대해서
	{
		inputbase =  batchindex * N * N * D1 + N * N * bi;
		weightbase = 3 * 3 * (dimensionindex * D1 + bi);
		for (k = 0; k < 3; k++) {
			for (l = 0; l < 3; l++) {
				x = hy + k - 1;
				y = hx + l - 1;
				if (x >= 0 && x < N && y >= 0 && y < N)
						sum += Input[inputbase + x * N + y] * Weight[weightbase+ k * 3 + l];
			}
		}
	}
	output[localindex]=ReLU(sum+B);
}
