#pragma once

#ifdef win32
#define SHARED_EXPORT extern "C" __declspec(dllexport)
#else
#define SHARED_EXPORT extern "C"
#endif
