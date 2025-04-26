#include "test.hpp"

int main()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
    ui(RED, 16);

    return 0;
}
