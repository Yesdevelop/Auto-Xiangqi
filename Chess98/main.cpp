#include "test.hpp"

int main()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
    testByUI();

    return 0;
}
