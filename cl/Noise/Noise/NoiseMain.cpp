// Copyright 2009-2015 Intel Corporation.
//
// The source code, information and material ("Material") contained herein is 
// owned by Intel Corporation or its suppliers or licensors, and title to such 
// Material remains with Intel Corporation or its suppliers or licensors. 
// The Material contains proprietary information of Intel or its suppliers and 
// licensors. The Material is protected by worldwide copyright laws and treaty 
// provisions. No part of the Material may be used, copied, reproduced, modified, 
// published, uploaded, posted, transmitted, distributed or disclosed in any way 
// without Intel's prior express written permission. No license under any patent, 
// copyright or other intellectual property rights in the Material is granted to 
// or conferred upon you, either expressly, by implication, inducement, estoppel 
// or otherwise. Any license under such intellectual property rights must be 
// express and approved by Intel in writing.
//
// *OpenCL(TM) - OpenCL and the OpenCL logo are trademarks of Apple Inc. used by 
// permission by Khronos.
//
// *Third Party trademarks are the property of their respective owners.
//
// Intel and the Intel logo are trademarks of Intel Corporation in the U.S. and/or 
// other countries.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter this 
// notice or any other notice embedded in Materials by Intel or Intel's suppliers 
// or licensors in any way.



#include "stdafx.h"


using namespace std;

#pragma warning( push )


// to enable kernel version with per pixel processing
#define PER_PIXEL



// declaration of native function
void    Noise2dTest ( float* outputArray, int width, int height, float slice );
void    Noise3dTest ( float* outputArray, int width, int height, float slice );
void    CloudTest   ( float* outputArray, int width, int height, float slice );
void    RandomTest  ( float* outputArray, int width, int height, float slice );



float ExecuteNoiseReference(int  testNum, cl_float* p_output, cl_uint width, cl_uint height, cl_float zSlice )
{
    double   perf_start;
    double   perf_stop;

    
    perf_start = time_stamp();
    {
        switch( testNum )
        {
        case    TEST_RANDOM:
                    RandomTest( p_output, width, height, zSlice);
                    break;

    //    case    TEST_2D_NOISE:
    //              Noise2dTest( p_output, width, height, zSlice);
    //              break;

        case    TEST_3D_NOISE:
                    Noise3dTest( p_output, width, height, zSlice);
                    break;

        case    TEST_CLOUD:
                    CloudTest( p_output, width, height, zSlice);
                    break;

        }
    }
    perf_stop = time_stamp();
    
    return (float)(perf_stop - perf_start);

}



float ExecuteNoiseKernel(int  test, cl_float* p_output, cl_uint width, cl_uint height, cl_float zSlice, OpenCLBasic& ocl )
{
    cl_int   err = CL_SUCCESS;
    double   perf_start;
    double   perf_stop;
    OpenCLProgramOneKernel *exec;

        // Build test opencl kernel
    
    switch( test )
    {
    //    case    TEST_2D_NOISE:
    //                printf("Running Noise 2d Test \n");
	//	            exec = new OpenCLProgramOneKernel( ocl, L"Noise.cl", "", "Noise2dTest", "-cl-fast-relaxed-math -cl-denorms-are-zero -g -s \"C:\\CRAVER\\PROJECTS\\intel_ocl_samples_win\\Perlin\\Noise.cl\" ");
    //                break;

        case    TEST_3D_NOISE:
                    printf("\n***Running Test: Perlin Noise \n");
		            exec = new OpenCLProgramOneKernel (ocl, L"Noise.cl", "", "Noise3dTest", "-cl-fast-relaxed-math -cl-denorms-are-zero -g -s \"C:\\CRAVER\\PROJECTS\\intel_ocl_samples_win\\Perlin\\Noise.cl\" ");	
                    break;

        case    TEST_CLOUD:
                    printf("\n***Running Test:  Cloud Generator \n");
		            exec = new OpenCLProgramOneKernel (ocl, L"Noise.cl", "", "CloudTest", "-cl-fast-relaxed-math -cl-denorms-are-zero -g -s \"C:\\CRAVER\\PROJECTS\\intel_ocl_samples_win\\Perlin\\Noise.cl\" ");	
                    break;

        case    TEST_RANDOM:
                    printf("\n***Running Test:  Random Number Generator \n");
		            exec = new OpenCLProgramOneKernel (ocl, L"Noise.cl", "", "RandomTest", "-cl-fast-relaxed-math -cl-denorms-are-zero -g -s \"C:\\CRAVER\\PROJECTS\\intel_ocl_samples_win\\Perlin\\Noise.cl\" ");	
                    break;
        default:
                    throw Error("Unknown Test ");
                    break;
    }

    
        // create OCL buffers
    cl_mem cl_output_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_float) * 4 * width * height, ocl.device),
            p_output,
            &err
        );
            SAMPLE_CHECK_ERRORS(err);
            if (cl_output_buffer == (cl_mem)0)
                throw Error("Failed to create Output Buffer!");


    // Execute kernel 
    //      note:  not timing the above operations - typically they are done once for multiple executions
    perf_start = time_stamp();
    {
	    cl_float	invWidth = 1.0f / width;
	    cl_float	invHeight = 1.0f / height;

	    int index = 0;

	    err = clSetKernelArg(exec->kernel, index++, sizeof(cl_mem), (void *)&cl_output_buffer);
            SAMPLE_CHECK_ERRORS(err);
	    err = clSetKernelArg(exec->kernel, index++, sizeof(cl_int), (void *)&width);
	        SAMPLE_CHECK_ERRORS(err);
	    err = clSetKernelArg(exec->kernel, index++, sizeof(cl_int), (void *)&height);
	        SAMPLE_CHECK_ERRORS(err);
	    err = clSetKernelArg(exec->kernel, index++, sizeof(cl_float), (void *)&invWidth);
	        SAMPLE_CHECK_ERRORS(err);
	    err = clSetKernelArg(exec->kernel, index++, sizeof(cl_float), (void *)&invHeight);
	        SAMPLE_CHECK_ERRORS(err);
	    err = clSetKernelArg(exec->kernel, index++, sizeof(cl_float), (void *)&zSlice);
	        SAMPLE_CHECK_ERRORS(err);

        size_t global_size[2] = {width,height};
        //printf("Global work size %dx%d\n", (int)global_size[0],(int)global_size[1]);


        err = clEnqueueNDRangeKernel(ocl.queue, exec->kernel, sizeof(global_size)/sizeof(size_t), NULL, global_size, NULL, 0, NULL, NULL);
            SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl.queue);
            SAMPLE_CHECK_ERRORS(err);

    }
    perf_stop = time_stamp();



    //map data back to the host to insure host has same output image as GPU
    void* tmp_ptr = clEnqueueMapBuffer(ocl.queue, cl_output_buffer, true, CL_MAP_READ, 0, sizeof(cl_float4) * width * height, 0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS(err);
        if(tmp_ptr!=p_output)
        {
            throw Error("clEnqueueMapBuffer failed to return original pointer");
        }

    // Need to Unmap back to GPU in order to release the GPU buffer object - host memory is not released. 
    err = clEnqueueUnmapMemObject(ocl.queue, cl_output_buffer, tmp_ptr, 0, NULL, NULL);
        SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(cl_output_buffer);
        SAMPLE_CHECK_ERRORS(err);

    return (float)(perf_stop - perf_start);
}



    // Main execution routine 
int main (int argc, const char** argv)
{
    int ret = EXIT_SUCCESS; //return code
    // pointer to the HOST buffers
    cl_float* p_output = NULL;
    cl_float* p_ref = NULL;
    try
    {
        cl_uint		width = 1600;	// -s --size:   width height of generated image
        cl_uint		height = 900;

        int         testNum = TEST_RANDOM;      // selector of which test to run, default
		cl_float	zSlice = 1.0f;	            // -z -- zslice : fixed "z" for 3d noise testing


        printf("Test options include: random, noise, and clouds\n");

        // Define and parse command-line arguments.
        CmdParserCommon cmd(argc, argv);
        CmdOptionErrors param_max_error_count(cmd);

        cmd.parse();

        // Immediatly exit if user wanted to see the usage information only.
        if(cmd.help.isSet())
        {
            return 0;
        }

        // Create the necessary OpenCL objects up to device queue.
        OpenCLBasic ocl(
            cmd.platform.getValue(),
            cmd.device_type.getValue(),
            cmd.device.getValue()
        );

        // special parsing for Noise specific options
        testNum = parseTest( cmd.run.getValue() );
        zSlice = (float) cmd.seed.getValue();


        // Create image buffers with alignment&size needed to avoid extra copies 
        //      on Intel integrated graphics processor OpenCL runtime
        //      (Do same for reference output, for equal treatment.)
        cl_uint     dev_alignment = zeroCopyPtrAlignment(ocl.device);
        size_t      aligned_size  = zeroCopySizeAlignment(sizeof(cl_float) * 4 * width * height, ocl.device);

        cl_float*   p_output = (cl_float*)aligned_malloc(aligned_size, dev_alignment);
        cl_float*   p_ref = (cl_float*)aligned_malloc(aligned_size, dev_alignment);



            //  printf("Executing OpenCL kernel...\n");
        float ocl_time = ExecuteNoiseKernel(testNum, p_output, width, height, zSlice, ocl );


            //  printf("Executing Reference Code\n");
        float ref_time = ExecuteNoiseReference(testNum, p_ref, width, height, zSlice );


        printf("   Opencl time %.3f ms,  Reference code time %.3f ms\n\n", ocl_time*1000.f, ref_time*1000.f);


        // Do verification
        //  printf("Performing verification...\n");
        int error_count = 0;
        for(cl_uint i = 0; i < width*height*4; i++)
        {
            // Compare the data
            if( fabsf(p_output[i] - p_ref[i]) > 0.1 )
            {
                printf("Error at location %d,  p_output = %f, p_ref = %f \n", i, p_output[i], p_ref[i]);
                error_count++;
                if(param_max_error_count.getValue()>0 && error_count>=param_max_error_count.getValue())
                    break;
            }
        }
        if(error_count)
        {
            printf("ERROR: Verification failed.\n");
            ret = EXIT_FAILURE;
        }
        else
        {
            printf("OpenCL / Reference Results Match.\n");
        }


        // save results in bitmap files
        SaveImageAsBMP_32FC4( p_output, 1.0f, width, height, "OpenCLOutput.bmp");
        SaveImageAsBMP_32FC4( p_ref, 1.0f, width, height, "ReferenceOutput.bmp");

    }
    catch(const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        ret = EXIT_FAILURE;
    }
    catch(const Error& error)
    {
        cerr << "[ ERROR ] Sample application specific error: " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch(const exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch(...)
    {
        cerr << "[ ERROR ] Unknown/internal error happened.\n";
        ret = EXIT_FAILURE;
    }

    aligned_free( p_ref );
    aligned_free( p_output );

	cout << "\nHit ENTER to finish.";
	cin.ignore();

    return ret;

}

#pragma warning( pop )
