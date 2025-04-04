#include "test.hpp"

int main()
{
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    ui(RED, 16);

    return 0;
}
