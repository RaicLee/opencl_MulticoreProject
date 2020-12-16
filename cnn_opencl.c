/*
2020/11~2020/12/7
Sejong University, Department of Computer Engineering 
16011105 Jeasung Lee/ OPENCL CNN ACCELERATE PROJECT HOST
*/
#include <CL/cl.h>
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 100
#define LOCALWORKSIZE 64
#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }


char* readSource(char* kernelPath) {

	cl_int status;
	FILE* fp;
	char* source;
	long int size;

	fp = fopen(kernelPath, "rb");
	if (!fp) {
		printf("Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if (status != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	if (size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}

	rewind(fp);

	source = (char*)malloc(size + 1);
	if (source == NULL) {
		printf("Error allocating space for the kernel source\n");
		exit(-1);
	}
	int i;
	for (i = 0; i < size + 1; i++) {
		source[i] = '\0';
	}

	fread(source, 1, size, fp);
	source[size] = '\0';

	return source;
}
char* source;

cl_int status;
cl_uint numPlatforms = 0;
cl_platform_id* platforms = NULL;
cl_uint numDevices = 0;
cl_device_id* devices = NULL;
cl_context context = NULL;
cl_command_queue cmdQueue;
cl_program program;
cl_kernel CONVK, POOLK, FCK;
cl_mem mw1_1, mb1_1, mw1_2, mb1_2;
cl_mem mw2_1, mb2_1, mw2_2, mb2_2;
cl_mem mw3_1, mb3_1, mw3_2, mb3_2, mw3_3, mb3_3;
cl_mem mw4_1, mb4_1, mw4_2, mb4_2, mw4_3, mb4_3;
cl_mem mw5_1, mb5_1, mw5_2, mb5_2, mw5_3, mb5_3;
cl_mem mw1	, mb1  , mw2  , mb2  , mw3  , mb3;
int globalWorkSize[3] = { 0,SIZE,1};
int localWorkSize[3] = { LOCALWORKSIZE,1,1};
void  checkProgramError(int err)
{
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char* log = (char*)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}
}
void cnn_init() {

	status = clGetPlatformIDs(0, NULL, &numPlatforms); CHECK_ERROR(status);
    platforms =(cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms,NULL); CHECK_ERROR(status);

    status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_ALL,0,NULL,&numDevices);
    devices =(cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

    status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_ALL,numDevices,devices,NULL); CHECK_ERROR(status);

    context = clCreateContext(NULL,numDevices,devices,NULL,NULL,&status); CHECK_ERROR(status);

    cmdQueue = clCreateCommandQueue(context,devices[0],0,&status); CHECK_ERROR(status);

	source = readSource("kernel.cl");
	program = clCreateProgramWithSource(context,1,(const char**)&source,NULL,&status); CHECK_ERROR(status);

	char option[80];
	sprintf(option, "-cl-fast-relaxed-math -D ReLU(x)=(((x)>0)?(x):0) -D MAX(a,b)=(a>b?a:b)");
	status = clBuildProgram(program,numDevices,devices,option,NULL,NULL); CHECK_ERROR(status);
	checkProgramError(status);
	POOLK = clCreateKernel(program, "pooling2x2_BATCH", &status); CHECK_ERROR(status);
	CONVK = clCreateKernel(program, "conv_BATCH", &status); CHECK_ERROR(status);
	FCK = clCreateKernel(program, "fc_layer_BATCH", &status); CHECK_ERROR(status);
}

void firstline(float** network)
{
	//first line
	/* index 0
	64 * 3 * 3 * 3,
	64,
	64 * 64 * 3 * 3,
	64,
	*/
	mw1_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		64 * 3 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw1_1,
		CL_FALSE,
		0,
		64 * 3 * 3 * 3 * sizeof(float),
		network[0],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb1_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		64 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb1_1,
		CL_FALSE,
		0,
		64 * sizeof(float),
		network[1],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	//2
	mw1_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		64 * 64 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw1_2,
		CL_FALSE,
		0,
		64 * 64 * 3 * 3 * sizeof(float),
		network[2],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb1_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		64 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb1_2,
		CL_FALSE,
		0,
		64 * sizeof(float),
		network[3],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
}
void secondline(float** network)
{
	/*
	* index 4
	128 * 64 * 3 * 3, 
	128,
	128 * 128 * 3 * 3,
	128,
	*/
	mw2_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		128 * 64 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw2_1,
		CL_FALSE,
		0,
		128 * 64 * 3 * 3 * sizeof(float),
		network[4],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
	mb2_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		128 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb2_1,
		CL_FALSE,
		0,
		128 * sizeof(float),
		network[5],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
	//2
	mw2_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		128 * 128 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw2_2,
		CL_FALSE,
		0,
		128 * 128 * 3 * 3 * sizeof(float),
		network[6],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
	mb2_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		128 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb2_2,
		CL_FALSE,
		0,
		128 * sizeof(float),
		network[7],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
}
void thirdline(float** network)
{
	/*
	* index 8
	256 * 128 * 3 * 3,
	256,
	256 * 256 * 3 * 3,
	256,
	256 * 256 * 3 * 3,
	256,
	*/
	mw3_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		256 * 128 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw3_1,
		CL_FALSE,
		0,
		256 * 128 * 3 * 3 * sizeof(float),
		network[8],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
	mb3_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		256 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb3_1,
		CL_FALSE,
		0,
		256 * sizeof(float),
		network[9],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	//2
	mw3_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		256 * 256 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw3_2,
		CL_FALSE,
		0,
		256 * 256 * 3 * 3 * sizeof(float),
		network[10],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
	mb3_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		256 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb3_2,
		CL_FALSE,
		0,
		256 * sizeof(float),
		network[11],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	//3
	mw3_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		256 * 256 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw3_3,
		CL_FALSE,
		0,
		256 * 256 * 3 * 3 * sizeof(float),
		network[12],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
	mb3_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		256 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb3_3,
		CL_FALSE,
		0,
		256 * sizeof(float),
		network[13],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

}
void fourthline(float** network)
{
	/*
	index 14
	512 * 256 * 3 * 3,
	512,
	512 * 512 * 3 * 3,
	512,
	512 * 512 * 3 * 3,
	512,
	*/
	mw4_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 256 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw4_1,
		CL_FALSE,
		0,
		512 * 256 * 3 * 3 * sizeof(float),
		network[14],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb4_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb4_1,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[15],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);


	//2

	mw4_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw4_2,
		CL_FALSE,
		0,
		512 * 512 * 3 * 3 * sizeof(float),
		network[16],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb4_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb4_2,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[17],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	//3
	mw4_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw4_3,
		CL_FALSE,
		0,
		512 * 512 * 3 * 3 * sizeof(float),
		network[18],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb4_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb4_3,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[19],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);


}
void fifthline(float** network)
{
	/*
	index 20
	512 * 512 * 3 * 3,
	512,
	512 * 512 * 3 * 3,
	512,
	512 * 512 * 3 * 3,
	512,
	*/

	mw5_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw5_1,
		CL_FALSE,
		0,
		512 * 512 * 3 * 3 * sizeof(float),
		network[20],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb5_1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb5_1,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[21],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);



	//2
	mw5_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw5_2,
		CL_FALSE,
		0,
		512 * 512 * 3 * 3 * sizeof(float),
		network[22],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb5_2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb5_2,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[23],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);


	//3
	mw5_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * 3 * 3 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw5_3,
		CL_FALSE,
		0,
		512 * 512 * 3 * 3 * sizeof(float),
		network[24],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb5_3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb5_3,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[25],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
}
void sixthline(float** network)
{
	/*
	index 26
	512 * 512,
	512,
	512 * 512,
	512,
	10 * 512,
	10
	*/
	mw1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw1,
		CL_FALSE,
		0,
		512 * 512 * sizeof(float),
		network[26],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb1,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[27],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);



	//2
	mw2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * 512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw2,
		CL_FALSE,
		0,
		512 * 512 * sizeof(float),
		network[28],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb2,
		CL_FALSE,
		0,
		512 * sizeof(float),
		network[29],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);


	//3
	mw3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		10 * 512 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mw3,
		CL_FALSE,
		0,
		10 * 512 * sizeof(float),
		network[30],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);

	mb3 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		10 * sizeof(float),
		NULL,
		&status);
	CHECK_ERROR(status);
	status = clEnqueueWriteBuffer(
		cmdQueue,
		mb3,
		CL_FALSE,
		0,
		10 * sizeof(float),
		network[31],
		0,
		NULL,
		NULL);
	CHECK_ERROR(status);
}
void networkInit(float** network)
{
	firstline(network);
	secondline(network);
	thirdline(network);
	fourthline(network);
	fifthline(network);
	sixthline(network);
}
void networkRelease()
{
	//mw1 release
	clReleaseMemObject(mw1_1);
	clReleaseMemObject(mw2_1);
	clReleaseMemObject(mw3_1);
	clReleaseMemObject(mw4_1);
	clReleaseMemObject(mw5_1);
	clReleaseMemObject(mw1);
	//mw2 release
	clReleaseMemObject(mw1_2);
	clReleaseMemObject(mw2_2);
	clReleaseMemObject(mw3_2);
	clReleaseMemObject(mw4_2);
	clReleaseMemObject(mw5_2);
	clReleaseMemObject(mw2);
	//mw3 release
	clReleaseMemObject(mw3_3);
	clReleaseMemObject(mw4_3);
	clReleaseMemObject(mw5_3);
	clReleaseMemObject(mw3);

	//mb1 release
	clReleaseMemObject(mb1_1);
	clReleaseMemObject(mb2_1);
	clReleaseMemObject(mb3_1);
	clReleaseMemObject(mb4_1);
	clReleaseMemObject(mb5_1);
	clReleaseMemObject(mb1);
	//mb2 release
	clReleaseMemObject(mb1_2);
	clReleaseMemObject(mb2_2);
	clReleaseMemObject(mb3_2);
	clReleaseMemObject(mb4_2);
	clReleaseMemObject(mb5_2);
	clReleaseMemObject(mb2);
	//mb3 release
	clReleaseMemObject(mb3_3);
	clReleaseMemObject(mb4_3);
	clReleaseMemObject(mb5_3);
	clReleaseMemObject(mb3);
}


static void pooling_layer_cl(cl_mem *inputs, cl_mem* outputs, int D, int N) {
	
	status = clSetKernelArg(POOLK,0,sizeof(cl_mem),inputs); 
	status |= clSetKernelArg(POOLK,1,sizeof(cl_mem),outputs); 
	status |= clSetKernelArg(POOLK, 2, sizeof(cl_mem), &D); 
	status |= clSetKernelArg(POOLK, 3, sizeof(cl_mem), &N); CHECK_ERROR(status);

	globalWorkSize[0] = D * N * N;//전체 해야하는 일들 하나하나씩
	localWorkSize[0] = LOCALWORKSIZE;
	globalWorkSize[0] = (globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0];
	globalWorkSize[1] = SIZE;
	localWorkSize[1] = 1;
	status = clEnqueueNDRangeKernel(cmdQueue, POOLK,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);CHECK_ERROR(status);
}

static void convolution_layer_cl(cl_mem* inputs, cl_mem* outputs, cl_mem filters, cl_mem biases, int D2, int D1, int N, int LSize) {

	status = clSetKernelArg(CONVK,0,sizeof(cl_mem),inputs); 
	status |= clSetKernelArg(CONVK,1,sizeof(cl_mem),outputs); 
	status |= clSetKernelArg(CONVK,2,sizeof(cl_mem),&filters); 
	status |= clSetKernelArg(CONVK,3,sizeof(cl_mem),&biases); 
	status |= clSetKernelArg(CONVK,4,sizeof(int),&N); 
	status |= clSetKernelArg(CONVK,5,sizeof(int),&D1);
	status |= clSetKernelArg(CONVK, 6, sizeof(int), &D2); CHECK_ERROR(status);
	globalWorkSize[0] = N * N;
	localWorkSize[0] = 16;
	globalWorkSize[0] = (globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0];
	globalWorkSize[1] = D2;
	localWorkSize[1] = 2*LSize;
	globalWorkSize[1] = (globalWorkSize[1] + localWorkSize[1] - 1) / localWorkSize[1] * localWorkSize[1];

	globalWorkSize[2] = SIZE;
	localWorkSize[2] = 1;
	status = clEnqueueNDRangeKernel(cmdQueue, CONVK,3,NULL, globalWorkSize, localWorkSize,0,NULL,NULL); CHECK_ERROR(status);

	globalWorkSize[1] = SIZE;

}

static void fc_layer_cl(cl_mem* input_neuron, cl_mem* output_neuron, cl_mem filters, cl_mem biases, int M, int N, int LSize) {

	status = clSetKernelArg(FCK,0,sizeof(cl_mem),input_neuron);
	status |= clSetKernelArg(FCK,1,sizeof(cl_mem),&filters);
	status |= clSetKernelArg(FCK,2,sizeof(cl_mem),&biases);
	status |= clSetKernelArg(FCK,3,sizeof(cl_mem),output_neuron);
	status |= clSetKernelArg(FCK,4,sizeof(int),&M); CHECK_ERROR(status); //OutputSize
	status |= clSetKernelArg(FCK, 5, sizeof(int), &N); CHECK_ERROR(status); // InputSIze

	globalWorkSize[0] = M;
	localWorkSize[0] = LOCALWORKSIZE;
	globalWorkSize[0] = (globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0];
	status = clEnqueueNDRangeKernel(cmdQueue, FCK,2,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
}

static void softmax(float* output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float* fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

float* alloc_layer(size_t n) {
	return (float*)calloc(n , sizeof(float));
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	
	networkInit(network);//네트워크 clBuffer에 넣기
	
	float * fc3;
	//크기 할당
	fc3 = alloc_layer(SIZE * 10);
	//cl_mem 버퍼만들기
	cl_mem IMAGE;
	IMAGE = clCreateBuffer(context,CL_MEM_READ_ONLY, SIZE * 32 * 32 * 3 * sizeof(float),NULL,&status); CHECK_ERROR(status);

	cl_mem mc1_1,mc1_2,mp1;
	mc1_1 = clCreateBuffer(context,CL_MEM_READ_WRITE, SIZE * 64 * 32 * 32 * sizeof(float),NULL,&status); CHECK_ERROR(status);
	mc1_2 = clCreateBuffer(context,CL_MEM_READ_WRITE, SIZE * 64 * 32 * 32 * sizeof(float),NULL,&status); CHECK_ERROR(status);
	mp1 = clCreateBuffer(context,CL_MEM_READ_WRITE, SIZE * 64 * 16 * 16 * sizeof(float),NULL,&status); CHECK_ERROR(status);

	cl_mem mc2_1, mc2_2, mp2;
	mc2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 128 * 16 * 16 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 128 * 16 * 16 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mp2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 128 * 8 * 8 * sizeof(float), NULL, &status); CHECK_ERROR(status);

	cl_mem mc3_1, mc3_2, mc3_3, mp3;
	mc3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 256 * 8 * 8 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 256 * 8 * 8 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 256 * 8 * 8 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mp3 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 256 * 4 * 4 * sizeof(float), NULL, &status); CHECK_ERROR(status);

	cl_mem mc4_1, mc4_2, mc4_3, mp4;
	mc4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 4 * 4 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 4 * 4 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 4 * 4 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mp4 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 2 * 2 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	cl_mem mc5_1, mc5_2, mc5_3, mp5;
	mc5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 2 * 2 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 2 * 2 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mc5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 2 * 2 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mp5 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * 1 * 1 * sizeof(float), NULL, &status); CHECK_ERROR(status);

	cl_mem mfc1, mfc2, mfc3;
	mfc1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mfc2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 512 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	mfc3 = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE * 10 * sizeof(float), NULL, &status); CHECK_ERROR(status);
	
	for (int i = 0; i < num_images/SIZE; ++i)
	{
		float* image = images + SIZE * i * 3 * 32 * 32;
		clEnqueueWriteBuffer(cmdQueue, IMAGE, CL_FALSE, 0, SIZE * 32 * 32 * 3 * sizeof(float), image, 0, NULL, NULL);
		
		convolution_layer_cl(&IMAGE, &mc1_1, mw1_1, mb1_1, 64, 3, 32,2);
		convolution_layer_cl(&mc1_1, &mc1_2, mw1_2, mb1_2, 64, 64, 32,2);
		pooling_layer_cl(&mc1_2,&mp1, 64, 16);
		
		convolution_layer_cl(&mp1, &mc2_1, mw2_1, mb2_1, 128, 64, 16,2);
		convolution_layer_cl(&mc2_1, &mc2_2, mw2_2, mb2_2, 128, 128, 16,2);		
		pooling_layer_cl(&mc2_2, &mp2, 128, 8);
		
		convolution_layer_cl(&mp2, &mc3_1, mw3_1, mb3_1, 256, 128, 8,4);
		convolution_layer_cl(&mc3_1, &mc3_2, mw3_2, mb3_2, 256, 256, 8,4);
		convolution_layer_cl(&mc3_2, &mc3_3, mw3_3, mb3_3, 256, 256, 8,4);
		pooling_layer_cl(&mc3_3, &mp3, 256, 4);
		
		convolution_layer_cl(&mp3, &mc4_1, mw4_1, mb4_1, 512, 256, 4,8);
		convolution_layer_cl(&mc4_1, &mc4_2, mw4_2, mb4_2, 512, 512, 4,8);
		convolution_layer_cl(&mc4_2, &mc4_3, mw4_3, mb4_3, 512, 512, 4,8);
		pooling_layer_cl(&mc4_3, &mp4, 512, 2);
	
		convolution_layer_cl(&mp4, &mc5_1, mw5_1, mb5_1, 512, 512, 2,8);
		convolution_layer_cl(&mc5_1, &mc5_2, mw5_2, mb5_2, 512, 512, 2,8);
		convolution_layer_cl(&mc5_2, &mc5_3, mw5_3, mb5_3, 512, 512, 2,8);
		pooling_layer_cl(&mc5_3, &mp5, 512, 1);
		
		fc_layer_cl(&mp5, &mfc1, mw1, mb1, 512, 512,64);
		fc_layer_cl(&mfc1, &mfc2, mw2, mb2, 512, 512,64);
		fc_layer_cl(&mfc2, &mfc3, mw3, mb3, 10, 512,2);

		clEnqueueReadBuffer(cmdQueue,mfc3,CL_TRUE,0, SIZE * 10 * sizeof(float),fc3,0,NULL,NULL);
		float* backup = fc3;
		for (int k = 0; k < SIZE; k++)
		{
			softmax(fc3, 10);
			labels[i*SIZE+k] = find_max(fc3, 10);
			confidences[i * SIZE + k] = fc3[labels[i * SIZE + k]];
			fc3= fc3 + 10;
		}
		fc3 = backup;
	}
	
	free(fc3);

	//1
	clReleaseMemObject(mc1_1);
	clReleaseMemObject(mc1_2);
	clReleaseMemObject(mp1);
	//2
	clReleaseMemObject(mc2_1);
	clReleaseMemObject(mc2_2);
	clReleaseMemObject(mp2);
	//3
	clReleaseMemObject(mc3_1);
	clReleaseMemObject(mc3_2);
	clReleaseMemObject(mc3_3);
	clReleaseMemObject(mp3);
	//4
	clReleaseMemObject(mc4_1);
	clReleaseMemObject(mc4_2);
	clReleaseMemObject(mc4_3);
	clReleaseMemObject(mp4);
	//5
	clReleaseMemObject(mc5_1);
	clReleaseMemObject(mc5_2);
	clReleaseMemObject(mc5_3);
	clReleaseMemObject(mp5);
	//6
	clReleaseMemObject(mfc1);
	clReleaseMemObject(mfc2);
	clReleaseMemObject(mfc3);
	//7
	networkRelease();
	//8
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);
	//9
	clReleaseKernel(POOLK);
	clReleaseKernel(CONVK);
	clReleaseKernel(FCK);
	//10
	free(platforms);
	free(devices);
}
