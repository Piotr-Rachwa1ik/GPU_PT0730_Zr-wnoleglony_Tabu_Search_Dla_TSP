#include <CL/cl.hpp>

#include <fstream>
#include <vector>
#include <sstream>

inline std::vector<cl::Device> list_devices() 
{
    std::vector<cl::Device> devices;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for(auto& platform : platforms) {
        std::vector<cl::Device> platform_devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
        if(platform_devices.size()==0) continue;
        for(auto& device : platform_devices) {
            devices.push_back(device);
        }
    }

    return devices;
}

inline std::string load_kernel(std::string fname)
{
    return (std::stringstream{} << std::ifstream{fname}.rdbuf()).str();
}