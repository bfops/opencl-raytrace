
******************************************************************************
**              Parallel Noise and Random Functions for OpenCL Kernels      **
**                                 README                                   **
******************************************************************************


*****  Overview  *****

This is an Intel(R) created *OpenCL sample source code, including an example 
of a random number generator and an implementation of the Perlin Noise algorithm.  
The Perlin noise algorithm is also demonstrated with a "cloud" image generator.  
The functions in the file Noise.cl are intended to be used by OpenCL kernels, 
and several sample tests of those functions are also included in Noise.cl.   
The associated C++ sample source code implements the "host" code portion 
required to execute the sample OpenCL kernels

The LICENSE.rtf file included in this distribution defines all conditions and 
limitations on use of all components of this distribution.


*****  Software Requirements  *****
The following software is required to correctly build and run samples:
  - Microsoft Visual Studio* 2010, 2012, or 2013
  - Visit  http://software.intel.com/en-us/opencl-code-builder to obtain the 
       Intel(R) Code Builder for OpenCL(tm) API
  - Intel OpenCL implementation for your target hardware - commonly distributed 
       as part of Intel graphics drivers.

This distribution includes project and solution files for *Microsoft Visual 
Studio 2012 and 2013   


***** Running Samples *****
For information on running the 'Noise' sample, build it and then run with the
following command line:

	Noise.exe  -h

The "--run" and "--seed" options are of particular interest - other options
may generally be left as defaults on systems with Intel HD Graphics.


*****  Disclaimer and Legal Information *****
The source code, information and material ("Material") contained herein is owned 
by Intel Corporation or its suppliers or licensors, and title to such Material 
remains with Intel Corporation or its suppliers or licensors. The Material contains 
proprietary information of Intel or its suppliers and licensors. The Material is 
protected by worldwide copyright laws and treaty provisions. No part of the Material 
may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, 
distributed or disclosed in any way without Intel's prior express written permission. 
No license under any patent, copyright or other intellectual property rights in the 
Material is granted to or conferred upon you, either expressly, by implication, 
inducement, estoppel or otherwise. Any license under such intellectual property 
rights must be express and approved by Intel in writing.

*OpenCL(TM) - OpenCL and the OpenCL logo are trademarks of Apple Inc. used by 
permission by Khronos.

*Third Party trademarks are the property of their respective owners.

Intel and the Intel logo are trademarks of Intel Corporation in the U.S. and/or 
other countries.

Unless otherwise agreed by Intel in writing, you may not remove or alter this notice 
or any other notice embedded in Materials by Intel or Intel's suppliers or licensors 
in any way.

Copyright (c) 2010-2015 Intel Corporation. All rights reserved.
